import React, { useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const landmarkerRef = useRef(null);
  const drawingUtilsRef = useRef(null);
  const objectDetectorRef = useRef(null);

  const animationRef = useRef(null);
  const streamRef = useRef(null);

  const lastUpdateRef = useRef(0);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const [status, setStatus] = useState(
    "Cargando IA..."
  );

  const [metrics, setMetrics] = useState({
    faces: 0,
    smile: 0,
    liveness: 0,
    emotion: "Neutral",
    skinTone: "Desconocido",
    hairTone: "Desconocido",
    distance: "Media",
    objects: [],
    verdict: "Sin rostro",
  });

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
        setStatus(
          "Cargando reconocimiento facial..."
        );

        const vision =
          await FilesetResolver.forVisionTasks(
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

              numFaces: 3,

              outputFaceBlendshapes: true,

              outputFacialTransformationMatrixes: true,
            }
          );

        setStatus(
          "Cargando detector de objetos..."
        );

        const objectModel =
          await cocoSsd.load();

        if (!cancelled) {
          landmarkerRef.current =
            faceLandmarker;

          objectDetectorRef.current =
            objectModel;

          setReady(true);

          setStatus(
            "Sistema listo."
          );
        }
      } catch (error) {
        console.error(error);

        setStatus(
          "Error cargando IA."
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
        "Cámara activa."
      );

      predictLoop();
    } catch (error) {
      console.error(error);

      setStatus(
        "No se pudo abrir la cámara."
      );
    }
  }

  function stopCamera() {
    if (animationRef.current) {
      cancelAnimationFrame(
        animationRef.current
      );
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) =>
          track.stop()
        );
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

    const ctx = canvas.getContext("2d", {
      willReadFrequently: true,
    });

    const faceLandmarker =
      landmarkerRef.current;

    const objectDetector =
      objectDetectorRef.current;

    const drawingUtils =
      drawingUtilsRef.current;

    async function render() {
      if (
        !video ||
        !canvas ||
        !faceLandmarker ||
        !objectDetector ||
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

      // VIDEO
      ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      // OBJETOS
      const objects =
        await objectDetector.detect(video);

      objects.forEach((obj) => {
        const [x, y, w, h] = obj.bbox;

        ctx.strokeStyle =
          "#00ffff";

        ctx.lineWidth = 3;

        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle =
          "#00ffff";

        ctx.font =
          "bold 16px Arial";

        ctx.fillText(
          `${obj.class} ${Math.round(
            obj.score * 100
          )}%`,
          x,
          y - 10
        );
      });

      // ROSTRO
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
        liveness: 0,
        emotion: "Neutral",
        skinTone: "Desconocido",
        hairTone: "Desconocido",
        distance: "Media",
        objects: [],
        verdict: "Sin rostro",
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
              "rgba(0,255,180,0.25)",

            lineWidth: 1,
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
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          {
            color: "#00ff99",

            lineWidth: 2,
          }
        );

        const smile =
          (
            getBlendScore(
              blend,
              "mouthSmileLeft"
            ) +
            getBlendScore(
              blend,
              "mouthSmileRight"
            )
          ) / 2;

        const jawOpen =
          getBlendScore(
            blend,
            "jawOpen"
          );

        let emotion = "Neutral";

        if (smile > 0.4)
          emotion = "Feliz";

        if (jawOpen > 0.45)
          emotion = "Sorprendido";

        const liveness = Math.min(
          100,
          Math.round(
            40 + smile * 60
          )
        );

        const xs = landmarks.map(
          (p) => p.x * canvas.width
        );

        const ys = landmarks.map(
          (p) => p.y * canvas.height
        );

        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);

        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        const faceWidth =
          maxX - minX;

        const faceHeight =
          maxY - minY;

        ctx.strokeStyle =
          "#00ff88";

        ctx.lineWidth = 3;

        ctx.strokeRect(
          minX,
          minY,
          faceWidth,
          faceHeight
        );

        let distance = "Lejos";

        if (faceWidth > 320)
          distance = "Muy cerca";
        else if (faceWidth > 180)
          distance = "Cerca";

        const facePoint =
          landmarks[4];

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

        if (smile > 0.45)
          verdict =
            "Sonrisa detectada";

        nextMetrics = {
          faces,

          smile: Math.round(
            smile * 100
          ),

          liveness,

          emotion,

          skinTone,

          hairTone,

          distance,

          objects: objects.map(
            (o) => o.class
          ),

          verdict,
        };

        // HUD
        ctx.fillStyle =
          "rgba(0,0,0,0.6)";

        ctx.fillRect(
          20,
          20,
          340,
          230
        );

        ctx.strokeStyle =
          "#00ffcc";

        ctx.strokeRect(
          20,
          20,
          340,
          230
        );

        ctx.fillStyle =
          "#00ff99";

        ctx.font =
          "bold 18px Arial";

        const hud = [
          `Estado: ${verdict}`,
          `Emoción: ${emotion}`,
          `Piel: ${skinTone}`,
          `Cabello: ${hairTone}`,
          `Distancia: ${distance}`,
          `Liveness: ${liveness}%`,
          `Objetos: ${objects
            .slice(0, 3)
            .map((o) => o.class)
            .join(", ")}`,
        ];

        hud.forEach((line, i) => {
          ctx.fillText(
            line,
            35,
            50 + i * 28
          );
        });
      }

      if (
        now - lastUpdateRef.current >
        100
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
    <main className="min-h-screen bg-slate-950 text-white flex items-center justify-center p-4">
      <section className="w-full max-w-7xl grid grid-cols-1 xl:grid-cols-[1.5fr_0.7fr] gap-5">
        <div className="rounded-3xl overflow-hidden border border-cyan-500/20 bg-slate-900 shadow-2xl relative">
          <canvas
            ref={canvasRef}
            className="w-full aspect-video bg-black"
          />

          <video
            ref={videoRef}
            className="hidden"
            playsInline
            muted
          />

          {!running && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/80">
              <div className="text-center p-8">
                <h1 className="text-5xl font-black text-cyan-300 mb-4">
                  FACE AI
                </h1>

                <p className="text-slate-300">
                  Reconocimiento facial +
                  objetos en tiempo real
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="rounded-3xl bg-slate-900 border border-cyan-500/20 p-5 shadow-2xl">
          <h2 className="text-3xl font-black text-cyan-300 mb-2">
            {metrics.verdict}
          </h2>

          <p className="text-slate-400 mb-6">
            {status}
          </p>

          <div className="grid grid-cols-2 gap-3">
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

            <Metric
              label="Emoción"
              value={metrics.emotion}
            />

            <Metric
              label="Distancia"
              value={metrics.distance}
            />
          </div>

          <div className="mt-6 flex flex-col gap-3">
            <button
              onClick={startCamera}
              disabled={!ready || running}
              className="rounded-2xl bg-cyan-400 hover:bg-cyan-300 transition-all text-slate-950 font-black px-4 py-3 disabled:opacity-40"
            >
              Iniciar cámara
            </button>

            <button
              onClick={stopCamera}
              disabled={!running}
              className="rounded-2xl bg-slate-800 hover:bg-slate-700 transition-all text-white font-black px-4 py-3 disabled:opacity-40"
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
      <p className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </p>

      <p className="text-lg sm:text-xl font-black text-cyan-300 mt-1 break-words">
        {value}
      </p>
    </div>
  );
}