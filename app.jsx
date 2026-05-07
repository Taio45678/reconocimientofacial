import React, { useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Cargando MediaPipe...");
  const [metrics, setMetrics] = useState({
    faces: 0,
    smile: 0,
    leftBlink: 0,
    rightBlink: 0,
    liveness: 0,
    verdict: "Sin rostro",
  });

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });

        if (!cancelled) {
          landmarkerRef.current = faceLandmarker;
          setReady(true);
          setStatus("Listo. Tocá iniciar cámara.");
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
        },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      const canvas = canvasRef.current;
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      setRunning(true);
      setStatus("Cámara activa. Detectando rostro en tiempo real...");
      predictLoop();
    } catch (error) {
      console.error(error);
      setStatus("No pude acceder a la cámara. En celular usá HTTPS y aceptá permisos.");
    }
  }

  function stopCamera() {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setRunning(false);
  }

  function getBlendScore(blendshapes, name) {
    const item = blendshapes?.categories?.find((c) => c.categoryName === name);
    return item ? item.score : 0;
  }

  function predictLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const faceLandmarker = landmarkerRef.current;
    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);

    function render() {
      if (!video || !canvas || !faceLandmarker || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(render);
        return;
      }

      const now = performance.now();
      const result = faceLandmarker.detectForVideo(video, now);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-canvas.width, 0);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const faces = result.faceLandmarks?.length || 0;
      let nextMetrics = {
        faces,
        smile: 0,
        leftBlink: 0,
        rightBlink: 0,
        liveness: 0,
        verdict: faces ? "Rostro detectado" : "Sin rostro",
      };

      if (faces > 0) {
        const landmarks = result.faceLandmarks[0];
        const blend = result.faceBlendshapes?.[0];

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          { color: "rgba(0, 255, 180, 0.22)", lineWidth: 1 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          { color: "#00ff99", lineWidth: 2 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          { color: "#00ff99", lineWidth: 2 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          { color: "#ffcc00", lineWidth: 2 }
        );

        const smileLeft = getBlendScore(blend, "mouthSmileLeft");
        const smileRight = getBlendScore(blend, "mouthSmileRight");
        const blinkLeft = getBlendScore(blend, "eyeBlinkLeft");
        const blinkRight = getBlendScore(blend, "eyeBlinkRight");

        const smile = (smileLeft + smileRight) / 2;
        const blink = Math.max(blinkLeft, blinkRight);

        // Prueba de vida simple: premia que haya rostro + expresión dinámica detectable.
        // No es seguridad real anti-spoofing, es una demo educativa.
        const liveness = Math.min(100, Math.round(40 + smile * 35 + blink * 25));

        let verdict = "Rostro detectado";
        if (liveness >= 70) verdict = "Prueba de vida probable";
        if (smile > 0.45) verdict = "Sonrisa detectada";
        if (blink > 0.45) verdict = "Parpadeo detectado";

        nextMetrics = {
          faces,
          smile: Math.round(smile * 100),
          leftBlink: Math.round(blinkLeft * 100),
          rightBlink: Math.round(blinkRight * 100),
          liveness,
          verdict,
        };
      }

      setMetrics(nextMetrics);
      animationRef.current = requestAnimationFrame(render);
    }

    render();
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-4">
      <section className="w-full max-w-6xl grid lg:grid-cols-[1.4fr_0.8fr] gap-5">
        <div className="rounded-3xl overflow-hidden bg-slate-900 shadow-2xl border border-slate-800 relative">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className="w-full h-auto block bg-black" />

          {!running && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-center p-8">
              <div>
                <h1 className="text-3xl md:text-5xl font-bold mb-3">
                  FaceID Demo
                </h1>
                <p className="text-slate-300 max-w-xl">
                  Detección facial en tiempo real con MediaPipe, webcam o cámara frontal del celular.
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="rounded-3xl bg-slate-900 border border-slate-800 p-6 shadow-2xl">
          <div className="mb-6">
            <p className="text-sm uppercase tracking-widest text-cyan-300 mb-2">
              Estado
            </p>
            <h2 className="text-2xl font-bold">{metrics.verdict}</h2>
            <p className="text-slate-400 mt-2">{status}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-6">
            <Metric label="Rostros" value={metrics.faces} />
            <Metric label="Sonrisa" value={`${metrics.smile}%`} />
            <Metric label="Ojo izq." value={`${metrics.leftBlink}%`} />
            <Metric label="Ojo der." value={`${metrics.rightBlink}%`} />
          </div>

          <div className="mb-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-300">Prueba de vida simple</span>
              <span>{metrics.liveness}%</span>
            </div>
            <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-150"
                style={{ width: `${metrics.liveness}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Demo educativa: no reemplaza un sistema real anti-suplantación.
            </p>
          </div>

          <div className="flex gap-3">
            <button
              onClick={startCamera}
              disabled={!ready || running}
              className="flex-1 rounded-2xl bg-cyan-400 text-slate-950 font-bold px-4 py-3 disabled:opacity-40"
            >
              Iniciar cámara
            </button>
            <button
              onClick={stopCamera}
              disabled={!running}
              className="flex-1 rounded-2xl bg-slate-800 text-white font-bold px-4 py-3 disabled:opacity-40"
            >
              Detener
            </button>
          </div>

          <div className="mt-6 text-sm text-slate-400 leading-relaxed">
            <p>
              Para Vercel: funciona con HTTPS. En celular, abrí el link publicado y aceptá permisos de cámara.
            </p>
          </div>
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-2xl bg-slate-950 border border-slate-800 p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}
