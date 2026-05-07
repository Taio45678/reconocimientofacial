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
  const drawingUtilsRef = useRef(null);

  const animationRef = useRef(null);
  const streamRef = useRef(null);

  const lastUpdateRef = useRef(0);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const [status, setStatus] = useState(
    "Cargando MediaPipe..."
  );

  const [metrics, setMetrics] = useState({
    faces: 0,
    smile: 0,
    leftBlink: 0,
    rightBlink: 0,
    liveness: 0,
    skinTone: "Desconocido",
    hairTone: "Desconocido",
    emotion: "Neutral",
    verdict: "Sin rostro",
  });

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const faceLandmarker =
          await FaceLandmarker.createFromOptions(
            vision,
            {
              baseOptions: {
                modelAssetPath:
                  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                delegate: "GPU",
              },

              runningMode: "VIDEO",

              numFaces: 1,

              outputFaceBlendshapes: true,

              outputFacialTransformationMatrixes: true,
            }
          );

        if (!cancelled) {
          landmarkerRef.current =
            faceLandmarker;

          setReady(true);

          setStatus(
            "Listo. Iniciá la cámara."
          );
        }
      } catch (error) {
        console.error(error);

        setStatus(
          "No se pudo cargar MediaPipe."
        );
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

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",

            width: {
              ideal: 1280,
            },

            height: {
              ideal: 720,
            },

            frameRate: {
              ideal: 30,
            },
          },

          audio: false,
        });

      streamRef.current = stream;

      const video = videoRef.current;

      video.srcObject = stream;

      await video.play();

      const canvas = canvasRef.current;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d", {
        willReadFrequently: true,
      });

      drawingUtilsRef.current =
        new DrawingUtils(ctx);

      setRunning(true);

      setStatus(
        "Cámara activa. Detectando rostro..."
      );

      predictLoop();
    } catch (error) {
      console.error(error);

      setStatus(
        "No pude acceder a la cámara."
      );
    }
  }

  function stopCamera() {
    if (animationRef.current) {
      cancelAnimationFrame(
        animationRef.current
      );

      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setRunning(false);
  }

  function getBlendScore(
    blendshapes,
    name
  ) {
    const item =
      blendshapes?.categories?.find(
        (c) => c.categoryName === name
      );

    return item ? item.score : 0;
  }

  function classifySkin(r, g, b) {
    const brightness =
      (r + g + b) / 3;

    if (brightness < 55)
      return "Muy oscura";

    if (brightness < 90)
      return "Oscura";

    if (brightness < 140)
      return "Morena";

    if (brightness < 190)
      return "Clara";

    return "Muy clara";
  }

  function classifyHair(r, g, b) {
    const brightness =
      (r + g + b) / 3;

    if (brightness < 50)
      return "Negro";

    if (brightness < 90)
      return "Castaño oscuro";

    if (brightness < 140)
      return "Castaño";

    if (brightness < 190)
      return "Rubio oscuro";

    return "Rubio";
  }

  function getAverageColor(
    ctx,
    x,
    y,
    size = 10
  ) {
    try {
      const data = ctx.getImageData(
        x,
        y,
        size,
        size
      ).data;

      let r = 0;
      let g = 0;
      let b = 0;
      let count = 0;

      for (
        let i = 0;
        i < data.length;
        i += 4
      ) {
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
        count++;
      }

      return {
        r: Math.round(r / count),
        g: Math.round(g / count),
        b: Math.round(b / count),
      };
    } catch {
      return {
        r: 0,
        g: 0,
        b: 0,
      };
    }
  }

  function predictLoop() {
    const video = videoRef.current;

    const canvas = canvasRef.current;

    const faceLandmarker =
      landmarkerRef.current;

    const drawingUtils =
      drawingUtilsRef.current;

    const ctx = canvas.getContext("2d", {
      willReadFrequently: true,
    });

    function render() {
      if (
        !video ||
        !canvas ||
        !faceLandmarker ||
        video.readyState < 2
      ) {
        animationRef.current =
          requestAnimationFrame(render);

        return;
      }

      const now = performance.now();

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      // VIDEO + IA EN UNA SOLA IMAGEN
      ctx.save();

      ctx.scale(-1, 1);

      ctx.drawImage(
        video,
        -canvas.width,
        0,
        canvas.width,
        canvas.height
      );

      ctx.restore();

      const result =
        faceLandmarker.detectForVideo(
          video,
          now
        );

      const faces =
        result.faceLandmarks?.length || 0;

      let nextMetrics = {
        faces,
        smile: 0,
        leftBlink: 0,
        rightBlink: 0,
        liveness: 0,
        skinTone: "Desconocido",
        hairTone: "Desconocido",
        emotion: "Neutral",
        verdict: faces
          ? "Rostro detectado"
          : "Sin rostro",
      };

      if (faces > 0) {
        const landmarks =
          result.faceLandmarks[0];

        const blend =
          result.faceBlendshapes?.[0];

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          {
            color:
              "rgba(0,255,180,0.18)",
            lineWidth: 1,
          }
        );

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          {
            color: "#00ff99",
            lineWidth: 2,
          }
        );

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          {
            color: "#00ff99",
            lineWidth: 2,
          }
        );

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          {
            color: "#ffcc00",
            lineWidth: 2,
          }
        );

        const smileLeft =
          getBlendScore(
            blend,
            "mouthSmileLeft"
          );

        const smileRight =
          getBlendScore(
            blend,
            "mouthSmileRight"
          );

        const blinkLeft =
          getBlendScore(
            blend,
            "eyeBlinkLeft"
          );

        const blinkRight =
          getBlendScore(
            blend,
            "eyeBlinkRight"
          );

        const jawOpen =
          getBlendScore(
            blend,
            "jawOpen"
          );

        const smile =
          (smileLeft + smileRight) / 2;

        const blink = Math.max(
          blinkLeft,
          blinkRight
        );

        const liveness = Math.min(
          100,
          Math.round(
            45 +
              smile * 30 +
              blink * 25
          )
        );

        let emotion = "Neutral";

        if (smile > 0.45)
          emotion = "Feliz";

        if (jawOpen > 0.45)
          emotion = "Sorprendido";

        // PUNTO DE PIEL
        const facePoint =
          landmarks[4];

        // PUNTO DE PELO
        const hairPoint =
          landmarks[10];

        const faceX = Math.floor(
          facePoint.x * canvas.width
        );

        const faceY = Math.floor(
          facePoint.y * canvas.height
        );

        const hairX = Math.floor(
          hairPoint.x * canvas.width
        );

        const hairY = Math.floor(
          hairPoint.y *
            canvas.height -
            40
        );

        const skinColor =
          getAverageColor(
            ctx,
            faceX,
            faceY,
            12
          );

        const hairColor =
          getAverageColor(
            ctx,
            hairX,
            hairY,
            12
          );

        const skinTone =
          classifySkin(
            skinColor.r,
            skinColor.g,
            skinColor.b
          );

        const hairTone =
          classifyHair(
            hairColor.r,
            hairColor.g,
            hairColor.b
          );

        let verdict =
          "Rostro detectado";

        if (liveness > 70)
          verdict =
            "Prueba de vida probable";

        if (smile > 0.45)
          verdict =
            "Sonrisa detectada";

        if (blink > 0.45)
          verdict =
            "Parpadeo detectado";

        nextMetrics = {
          faces,

          smile: Math.round(
            smile * 100
          ),

          leftBlink: Math.round(
            blinkLeft * 100
          ),

          rightBlink: Math.round(
            blinkRight * 100
          ),

          liveness,

          skinTone,

          hairTone,

          emotion,

          verdict,
        };

        // HUD FUTURISTA
        ctx.fillStyle =
          "rgba(0,0,0,0.55)";

        ctx.fillRect(20, 20, 310, 190);

        ctx.strokeStyle =
          "rgba(0,255,180,0.5)";

        ctx.strokeRect(20, 20, 310, 190);

        ctx.fillStyle = "#00ff99";

        ctx.font =
          "bold 18px Arial";

        ctx.fillText(
          `Estado: ${verdict}`,
          35,
          50
        );

        ctx.fillText(
          `Emoción: ${emotion}`,
          35,
          80
        );

        ctx.fillText(
          `Piel: ${skinTone}`,
          35,
          110
        );

        ctx.fillText(
          `Cabello: ${hairTone}`,
          35,
          140
        );

        ctx.fillText(
          `Liveness: ${liveness}%`,
          35,
          170
        );
      }

      if (
        now - lastUpdateRef.current >
        120
      ) {
        setMetrics(nextMetrics);

        lastUpdateRef.current = now;
      }

      animationRef.current =
        requestAnimationFrame(render);
    }

    render();
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-4 overflow-hidden">
      <section className="w-full max-w-7xl grid lg:grid-cols-[1.4fr_0.8fr] gap-5">
        <div className="rounded-3xl overflow-hidden bg-slate-900 shadow-2xl border border-cyan-500/20 relative">

          {/* SOLO UN CANVAS */}
          <canvas
            ref={canvasRef}
            className="w-full h-auto block bg-black rounded-3xl"
          />

          {/* VIDEO OCULTO */}
          <video
            ref={videoRef}
            className="hidden"
            playsInline
            muted
          />

          {!running && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-center p-8">
              <div>
                <h1 className="text-4xl md:text-6xl font-black mb-4 text-cyan-300">
                  FACE AI
                </h1>

                <p className="text-slate-300 max-w-xl text-lg">
                  Reconocimiento facial
                  avanzado con MediaPipe.
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="rounded-3xl bg-slate-900 border border-cyan-500/20 p-6 shadow-2xl">
          <div className="mb-6">
            <p className="text-sm uppercase tracking-widest text-cyan-300 mb-2">
              Estado
            </p>

            <h2 className="text-3xl font-black">
              {metrics.verdict}
            </h2>

            <p className="text-slate-400 mt-2">
              {status}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-6">
            <Metric
              label="Rostros"
              value={metrics.faces}
            />

            <Metric
              label="Sonrisa"
              value={`${metrics.smile}%`}
            />

            <Metric
              label="Piel"
              value={metrics.skinTone}
            />

            <Metric
              label="Cabello"
              value={metrics.hairTone}
            />
          </div>

          <div className="mb-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-300">
                Prueba de vida
              </span>

              <span>
                {metrics.liveness}%
              </span>
            </div>

            <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-150"
                style={{
                  width: `${metrics.liveness}%`,
                }}
              />
            </div>

            <p className="text-sm text-slate-400 mt-4">
              Emoción detectada:
            </p>

            <h3 className="text-2xl font-bold text-cyan-300 mt-1">
              {metrics.emotion}
            </h3>
          </div>

          <div className="flex gap-3">
            <button
              onClick={startCamera}
              disabled={!ready || running}
              className="flex-1 rounded-2xl bg-cyan-400 hover:bg-cyan-300 transition-all text-slate-950 font-black px-4 py-3 disabled:opacity-40"
            >
              Iniciar
            </button>

            <button
              onClick={stopCamera}
              disabled={!running}
              className="flex-1 rounded-2xl bg-slate-800 hover:bg-slate-700 transition-all text-white font-black px-4 py-3 disabled:opacity-40"
            >
              Detener
            </button>
          </div>
        </aside>
      </section>
    </main>
  );
}

function Metric({
  label,
  value,
}) {
  return (
    <div className="rounded-2xl bg-slate-950 border border-slate-800 p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wider">
        {label}
      </p>

      <p className="text-xl font-black mt-1 text-cyan-300 break-words">
        {value}
      </p>
    </div>
  );
}