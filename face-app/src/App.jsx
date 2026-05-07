import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
  ObjectDetector,
} from "@mediapipe/tasks-vision";

const OBJECT_MAP = {
  book: { label: "Cuaderno / libro", type: "directa" },
  "cell phone": { label: "Celular", type: "directa" },
  cup: { label: "Mate / vaso", type: "aprox." },
  bottle: { label: "Termo / botella", type: "aprox." },
};

const UNSUPPORTED_OBJECTS = ["barbijo", "lente", "lapicera"];

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const objectDetectorRef = useRef(null);
  const drawingUtilsRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const lastUiUpdateRef = useRef(0);
  const lastObjectDetectRef = useRef(0);
  const lastObjectsRef = useRef([]);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Inicializando módulos IA...");
  const [metrics, setMetrics] = useState({
    faces: 0,
    emotion: "Sin rostro",
    emotionScore: 0,
    smile: 0,
    surprise: 0,
    blink: 0,
    focus: 0,
    liveness: 0,
    verdict: "Sin rostro",
    objects: [],
  });

  const objectBadges = useMemo(() => {
    const found = new Set(metrics.objects.map((o) => o.label));
    return [
      { label: "Cuaderno", active: [...found].some((x) => x.includes("Cuaderno")) },
      { label: "Celular", active: [...found].some((x) => x.includes("Celular")) },
      { label: "Mate", active: [...found].some((x) => x.includes("Mate")) },
      { label: "Termo", active: [...found].some((x) => x.includes("Termo")) },
      { label: "Barbijo*", active: false },
      { label: "Lente*", active: false },
      { label: "Lapicera*", active: false },
    ];
  }, [metrics.objects]);

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const [faceLandmarker, objectDetector] = await Promise.all([
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
          }),
          ObjectDetector.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            scoreThreshold: 0.38,
            maxResults: 8,
          }),
        ]);

        if (!cancelled) {
          faceLandmarkerRef.current = faceLandmarker;
          objectDetectorRef.current = objectDetector;
          setReady(true);
          setStatus("Listo. Iniciá la cámara.");
        }
      } catch (error) {
        console.error(error);
        setStatus("No se pudo cargar MediaPipe. Revisá conexión o consola.");
      }
    }

    initMediaPipe();
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, []);

  async function startCamera() {
    try {
      if (!ready) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      drawingUtilsRef.current = new DrawingUtils(ctx);

      setRunning(true);
      setStatus("Escaneo activo: rostro + objetos.");
      predictLoop();
    } catch (error) {
      console.error(error);
      setStatus("No pude acceder a la cámara. En celular usá HTTPS y aceptá permisos.");
    }
  }

  function stopCamera() {
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }

  function blend(blendshapes, name) {
    return blendshapes?.categories?.find((c) => c.categoryName === name)?.score || 0;
  }

  function classifyEmotion(faceBlend) {
    const smile = (blend(faceBlend, "mouthSmileLeft") + blend(faceBlend, "mouthSmileRight")) / 2;
    const frown = (blend(faceBlend, "mouthFrownLeft") + blend(faceBlend, "mouthFrownRight")) / 2;
    const jawOpen = blend(faceBlend, "jawOpen");
    const browUp = blend(faceBlend, "browInnerUp");
    const browDown = (blend(faceBlend, "browDownLeft") + blend(faceBlend, "browDownRight")) / 2;
    const eyeWide = (blend(faceBlend, "eyeWideLeft") + blend(faceBlend, "eyeWideRight")) / 2;
    const eyeSquint = (blend(faceBlend, "eyeSquintLeft") + blend(faceBlend, "eyeSquintRight")) / 2;
    const blinkLeft = blend(faceBlend, "eyeBlinkLeft");
    const blinkRight = blend(faceBlend, "eyeBlinkRight");
    const blinkMax = Math.max(blinkLeft, blinkRight);
    const blinkAvg = (blinkLeft + blinkRight) / 2;

    const scores = [
      { name: "Feliz", score: smile * 0.78 + eyeSquint * 0.22 },
      { name: "Sorprendido", score: jawOpen * 0.48 + browUp * 0.32 + eyeWide * 0.2 },
      { name: "Concentrado", score: browDown * 0.55 + eyeSquint * 0.25 + (1 - jawOpen) * 0.2 },
      { name: "Serio / neutral", score: Math.max(0.15, 0.55 - smile * 0.35 - jawOpen * 0.2 - browUp * 0.1) },
      { name: "Posible disgusto", score: frown * 0.55 + browDown * 0.25 + blend(faceBlend, "mouthPressLeft") * 0.1 + blend(faceBlend, "mouthPressRight") * 0.1 },
    ].sort((a, b) => b.score - a.score);

    const top = scores[0];
    const confidence = Math.round(Math.min(100, Math.max(0, top.score * 100)));

    return {
      emotion: confidence < 28 ? "Indefinida" : top.name,
      emotionScore: confidence,
      smile: Math.round(smile * 100),
      surprise: Math.round((jawOpen * 0.55 + browUp * 0.3 + eyeWide * 0.15) * 100),
      blink: Math.round(blinkMax * 100),
      focus: Math.round(Math.min(100, (1 - blinkAvg) * 55 + browDown * 35 + eyeSquint * 10)),
    };
  }

  function getTargetObjects(objectResult) {
    const detections = objectResult?.detections || [];
    return detections
      .map((det) => {
        const cat = det.categories?.[0];
        const key = cat?.categoryName;
        if (!key || !OBJECT_MAP[key]) return null;
        const mapped = OBJECT_MAP[key];
        return {
          key,
          label: mapped.label,
          type: mapped.type,
          score: Math.round((cat.score || 0) * 100),
          box: det.boundingBox,
        };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);
  }

  function drawObjectBoxes(ctx, objects) {
    objects.forEach((obj) => {
      const box = obj.box;
      if (!box) return;
      ctx.save();
      ctx.strokeStyle = obj.type === "directa" ? "#00f5ff" : "#fcee09";
      ctx.lineWidth = 3;
      ctx.shadowColor = ctx.strokeStyle;
      ctx.shadowBlur = 10;
      ctx.strokeRect(box.originX, box.originY, box.width, box.height);

      const text = `${obj.label} ${obj.score}% ${obj.type === "aprox." ? "≈" : ""}`;
      ctx.font = "bold 15px ui-monospace, SFMono-Regular, Menlo, monospace";
      const w = ctx.measureText(text).width + 14;
      ctx.fillStyle = "rgba(0,0,0,0.78)";
      ctx.fillRect(box.originX, Math.max(0, box.originY - 28), w, 24);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillText(text, box.originX + 7, Math.max(17, box.originY - 10));
      ctx.restore();
    });
  }

  function drawCyberOverlay(ctx, canvas, nextMetrics) {
    ctx.save();
    const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    grad.addColorStop(0, "rgba(0,245,255,0.08)");
    grad.addColorStop(0.5, "rgba(255,0,128,0.06)");
    grad.addColorStop(1, "rgba(252,238,9,0.05)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "rgba(0,245,255,0.16)";
    ctx.lineWidth = 1;
    const step = 64;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(2,6,23,0.72)";
    ctx.strokeStyle = "rgba(0,245,255,0.55)";
    ctx.lineWidth = 2;
    ctx.fillRect(18, 18, 330, 168);
    ctx.strokeRect(18, 18, 330, 168);
    ctx.font = "bold 17px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#00f5ff";
    ctx.fillText(`FACE: ${nextMetrics.verdict}`, 34, 50);
    ctx.fillStyle = "#fcee09";
    ctx.fillText(`EMOTION: ${nextMetrics.emotion} ${nextMetrics.emotionScore}%`, 34, 82);
    ctx.fillStyle = "#ff0080";
    ctx.fillText(`LIVE: ${nextMetrics.liveness}%`, 34, 114);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`OBJ: ${nextMetrics.objects.length}`, 34, 146);
    ctx.restore();
  }

  function predictLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const faceLandmarker = faceLandmarkerRef.current;
    const objectDetector = objectDetectorRef.current;
    const drawingUtils = drawingUtilsRef.current;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    function render() {
      if (!video || !canvas || !faceLandmarker || !objectDetector || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }

      const now = performance.now();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Sin modo espejo: se dibuja igual que viene de la cámara.
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const faceResult = faceLandmarker.detectForVideo(video, now);

      if (now - lastObjectDetectRef.current > 260) {
        try {
          const objectResult = objectDetector.detectForVideo(video, now);
          lastObjectsRef.current = getTargetObjects(objectResult);
          lastObjectDetectRef.current = now;
        } catch (error) {
          console.warn("Object detection skipped", error);
        }
      }

      const objects = lastObjectsRef.current;
      drawObjectBoxes(ctx, objects);

      const faces = faceResult.faceLandmarks?.length || 0;
      let nextMetrics = {
        faces,
        emotion: faces ? "Analizando" : "Sin rostro",
        emotionScore: 0,
        smile: 0,
        surprise: 0,
        blink: 0,
        focus: 0,
        liveness: 0,
        verdict: faces ? "Rostro detectado" : "Sin rostro",
        objects,
      };

      if (faces > 0) {
        const landmarks = faceResult.faceLandmarks[0];
        const faceBlend = faceResult.faceBlendshapes?.[0];

        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
          color: "rgba(0,245,255,0.18)",
          lineWidth: 1,
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
          color: "#ff0080",
          lineWidth: 2,
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, {
          color: "#00f5ff",
          lineWidth: 2,
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, {
          color: "#00f5ff",
          lineWidth: 2,
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
          color: "#fcee09",
          lineWidth: 2,
        });

        const emotion = classifyEmotion(faceBlend);
        const dynamicSignal = Math.max(emotion.smile, emotion.blink, emotion.surprise);
        const liveness = Math.min(100, Math.round(45 + dynamicSignal * 0.55));

        let verdict = "Rostro detectado";
        if (liveness >= 72) verdict = "Vida probable";
        if (emotion.blink > 45) verdict = "Parpadeo detectado";
        if (emotion.smile > 45) verdict = "Sonrisa detectada";

        nextMetrics = {
          faces,
          ...emotion,
          liveness,
          verdict,
          objects,
        };
      }

      drawCyberOverlay(ctx, canvas, nextMetrics);

      if (now - lastUiUpdateRef.current > 120) {
        setMetrics(nextMetrics);
        lastUiUpdateRef.current = now;
      }

      animationRef.current = requestAnimationFrame(render);
    }

    render();
  }

  return (
    <main className="min-h-dvh overflow-hidden bg-[#050816] text-white">
      <div className="fixed inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(0,245,255,0.22),transparent_28%),radial-gradient(circle_at_85%_12%,rgba(255,0,128,0.2),transparent_30%),linear-gradient(135deg,#050816,#09021a_55%,#030712)]" />
      <div className="fixed inset-0 opacity-30 [background-image:linear-gradient(rgba(0,245,255,.15)_1px,transparent_1px),linear-gradient(90deg,rgba(0,245,255,.15)_1px,transparent_1px)] [background-size:42px_42px]" />

      <section className="relative z-10 min-h-dvh w-full max-w-7xl mx-auto p-3 sm:p-5 grid grid-cols-1 lg:grid-cols-[1.45fr_0.75fr] gap-4 items-center">
        <div className="relative rounded-[28px] overflow-hidden border border-cyan-300/35 bg-black shadow-[0_0_45px_rgba(0,245,255,.16)] min-h-[58dvh] sm:min-h-0">
          <canvas ref={canvasRef} className="block w-full h-[62dvh] sm:h-auto object-cover bg-black" />
          <video ref={videoRef} className="hidden" playsInline muted />

          <div className="pointer-events-none absolute inset-0 bg-[repeating-linear-gradient(to_bottom,rgba(255,255,255,.045)_0px,rgba(255,255,255,.045)_1px,transparent_2px,transparent_5px)] mix-blend-overlay" />

          <div className="absolute top-3 left-3 right-3 flex items-center justify-between gap-3">
            <div className="rounded-full border border-cyan-300/50 bg-black/65 px-3 py-1 text-[10px] sm:text-xs tracking-[0.28em] text-cyan-200 backdrop-blur">
              CYBER FACE / OBJECT AI
            </div>
            <div className={`h-3 w-3 rounded-full ${running ? "bg-cyan-300 shadow-[0_0_18px_#00f5ff]" : "bg-pink-500 shadow-[0_0_18px_#ff0080]"}`} />
          </div>

          {!running && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/78 text-center p-6">
              <div>
                <p className="text-cyan-300 tracking-[0.35em] text-xs mb-3">MEDIAPIPE REAL TIME</p>
                <h1 className="text-5xl sm:text-7xl font-black leading-none text-white [text-shadow:0_0_12px_#00f5ff,0_0_30px_#ff0080]">
                  FACE<br />SCAN
                </h1>
                <p className="mt-4 text-slate-300 max-w-md mx-auto text-sm sm:text-base">
                  Rostro, emoción heurística, prueba de vida y objetos cercanos. Sin modo espejo.
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="rounded-[28px] border border-pink-400/35 bg-black/62 backdrop-blur p-4 sm:p-6 shadow-[0_0_45px_rgba(255,0,128,.14)]">
          <div className="mb-5">
            <p className="text-xs uppercase tracking-[0.35em] text-pink-300 mb-2">Estado</p>
            <h2 className="text-2xl sm:text-3xl font-black text-white [text-shadow:0_0_14px_#00f5ff]">{metrics.verdict}</h2>
            <p className="text-cyan-100/70 mt-2 text-sm">{status}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-5">
            <Metric label="Rostros" value={metrics.faces} />
            <Metric label="Emoción" value={`${metrics.emotion} ${metrics.emotionScore}%`} />
            <Metric label="Sonrisa" value={`${metrics.smile}%`} />
            <Metric label="Sorpresa" value={`${metrics.surprise}%`} />
            <Metric label="Parpadeo" value={`${metrics.blink}%`} />
            <Metric label="Foco" value={`${metrics.focus}%`} />
          </div>

          <div className="rounded-2xl border border-cyan-300/25 bg-slate-950/75 p-4 mb-5">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-cyan-100/80">Liveness demo</span>
              <span className="text-yellow-200 font-black">{metrics.liveness}%</span>
            </div>
            <div className="h-4 rounded-full bg-slate-900 overflow-hidden border border-cyan-300/20">
              <div className="h-full bg-gradient-to-r from-cyan-300 via-pink-500 to-yellow-300 transition-all duration-150" style={{ width: `${metrics.liveness}%` }} />
            </div>
          </div>

          <div className="mb-5">
            <p className="text-xs uppercase tracking-[0.3em] text-yellow-200 mb-3">Objetos objetivo</p>
            <div className="flex flex-wrap gap-2">
              {objectBadges.map((item) => (
                <span key={item.label} className={`rounded-full px-3 py-1 text-xs border ${item.active ? "border-cyan-300 bg-cyan-300/18 text-cyan-100" : "border-slate-700 bg-slate-950/70 text-slate-400"}`}>
                  {item.label}
                </span>
              ))}
            </div>
            <div className="mt-3 space-y-2">
              {metrics.objects.length === 0 ? (
                <p className="text-sm text-slate-400">Sin objetos objetivo detectados.</p>
              ) : (
                metrics.objects.map((o, i) => (
                  <div key={`${o.label}-${i}`} className="flex justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/70 px-3 py-2 text-sm">
                    <span>{o.label} <span className="text-slate-500">{o.type}</span></span>
                    <b className="text-cyan-200">{o.score}%</b>
                  </div>
                ))
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-3">
              * Barbijo, lente y lapicera requieren modelo custom para detección real. Este demo detecta book/cell phone/cup/bottle y los traduce a los objetos pedidos cuando corresponde.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <button onClick={startCamera} disabled={!ready || running} className="rounded-2xl bg-cyan-300 text-slate-950 font-black px-4 py-4 disabled:opacity-40 active:scale-[.98] shadow-[0_0_24px_rgba(0,245,255,.28)]">
              INICIAR
            </button>
            <button onClick={stopCamera} disabled={!running} className="rounded-2xl bg-pink-500/20 border border-pink-300/45 text-pink-100 font-black px-4 py-4 disabled:opacity-40 active:scale-[.98]">
              DETENER
            </button>
          </div>
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-2xl bg-slate-950/80 border border-cyan-300/20 p-3">
      <p className="text-[10px] text-cyan-100/50 uppercase tracking-[0.22em]">{label}</p>
      <p className="text-lg sm:text-xl font-black mt-1 text-cyan-100 break-words">{value}</p>
    </div>
  );
}