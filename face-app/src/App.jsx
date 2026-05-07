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

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function initMediaPipe() {
      try {
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

              numFaces: 5,

              outputFaceBlendshapes: true,

              outputFacialTransformationMatrixes: true,

              minFaceDetectionConfidence: 0.7,

              minFacePresenceConfidence: 0.7,

              minTrackingConfidence: 0.7,
            }
          );

        const objectModel =
          await cocoSsd.load({
            base: "mobilenet_v2",
          });

        if (!cancelled) {
          landmarkerRef.current =
            faceLandmarker;

          objectDetectorRef.current =
            objectModel;

          setReady(true);
        }
      } catch (error) {
        console.error(error);
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
              ideal: 60,
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

      predictLoop();
    } catch (error) {
      console.error(error);
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

      ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      // SCANNER FX
      const scanY =
        (Date.now() / 4) %
        canvas.height;

      ctx.fillStyle =
        "rgba(0,255,255,0.05)";

      ctx.fillRect(
        0,
        scanY,
        canvas.width,
        3
      );

      // OBJETOS
      const objects =
        await objectDetector.detect(video);

      objects
        .filter((obj) => obj.score > 0.6)
        .forEach((obj) => {
          const [x, y, w, h] = obj.bbox;

          ctx.strokeStyle =
            "#00f7ff";

          ctx.lineWidth = 3;

          ctx.shadowColor =
            "#00f7ff";

          ctx.shadowBlur = 15;

          ctx.strokeRect(x, y, w, h);

          ctx.fillStyle =
            "rgba(0,0,0,0.75)";

          ctx.fillRect(
            x,
            y - 34,
            180,
            30
          );

          ctx.fillStyle =
            "#00f7ff";

          ctx.font =
            window.innerWidth < 768
              ? "bold 12px Arial"
              : "bold 16px Arial";

          ctx.fillText(
            `${obj.class.toUpperCase()} ${Math.round(
              obj.score * 100
            )}%`,
            x + 10,
            y - 12
          );
        });

      // ROSTROS
      const result =
        faceLandmarker.detectForVideo(
          video,
          now
        );

      const faces =
        result.faceLandmarks || [];

      faces.forEach((landmarks, index) => {
        const blend =
          result.faceBlendshapes?.[index];

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          {
            color:
              "rgba(0,255,180,0.12)",

            lineWidth: 1,
          }
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

        // BOX
        ctx.strokeStyle =
          "#00ff99";

        ctx.lineWidth = 3;

        ctx.shadowColor =
          "#00ff99";

        ctx.shadowBlur = 18;

        ctx.strokeRect(
          minX,
          minY,
          faceWidth,
          faceHeight
        );

        // RADAR
        ctx.beginPath();

        ctx.arc(
          maxX + 30,
          minY + 30,
          18,
          0,
          Math.PI * 2
        );

        ctx.strokeStyle =
          "#00ffee";

        ctx.stroke();

        ctx.beginPath();

        ctx.arc(
          maxX + 30,
          minY + 30,
          5,
          0,
          Math.PI * 2
        );

        ctx.fillStyle =
          "#00ffee";

        ctx.fill();

        // DATOS
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

        const blinking =
          blinkLeft > 0.6 ||
          blinkRight > 0.6;

        let emotion = "Neutral";

        if (smile > 0.45)
          emotion = "Feliz";

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

        // HUD
        const hudWidth =
          window.innerWidth < 768
            ? 240
            : 320;

        const hudHeight =
          165;

        let hudY =
          minY - hudHeight - 12;

        if (hudY < 10) {
          hudY = minY + faceHeight + 12;
        }

        ctx.fillStyle =
          "rgba(0,0,0,0.68)";

        ctx.fillRect(
          minX,
          hudY,
          hudWidth,
          hudHeight
        );

        ctx.strokeStyle =
          "#00ffee";

        ctx.lineWidth = 2;

        ctx.strokeRect(
          minX,
          hudY,
          hudWidth,
          hudHeight
        );

        ctx.fillStyle =
          "#00ffee";

        ctx.font =
          window.innerWidth < 768
            ? "bold 11px Arial"
            : "bold 15px Arial";

        const info = [
          `FACE ID: ${index + 1}`,
          `EMOCION: ${emotion}`,
          `PIEL: ${skinTone}`,
          `CABELLO: ${hairTone}`,
          `PERSONAS: ${faces.length}`,
          blinking
            ? "PARPADEO DETECTADO"
            : "OJOS ABIERTOS",
        ];

        info.forEach((line, i) => {
          ctx.fillText(
            line,
            minX + 12,
            hudY + 24 + i * 22
          );
        });

        ctx.fillStyle =
          blinking
            ? "#ff4444"
            : "#00ff88";

        ctx.fillText(
          blinking
            ? "LIVE BLINK"
            : "TRACKING ACTIVE",
          minX + 12,
          hudY + hudHeight - 14
        );
      });

      animationRef.current =
        requestAnimationFrame(render);
    }

    render();
  }

  return (
    <main className="fixed inset-0 w-screen h-screen overflow-hidden bg-black">
      {/* FX */}
      <div
        className="absolute inset-0 opacity-[0.08] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(0,255,255,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,255,0.15) 1px, transparent 1px)",
          backgroundSize:
            "40px 40px",
        }}
      />

      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
      />

      <video
        ref={videoRef}
        className="hidden"
        playsInline
        muted
      />

      {/* TOP BAR */}
      <div className="absolute top-0 left-0 w-full flex items-center justify-between px-4 md:px-8 py-4 z-50">
        <div>
          <h1 className="text-cyan-300 font-black tracking-[0.25em] text-xs sm:text-sm md:text-xl">
            FACE AI SYSTEM
          </h1>

          <p className="text-cyan-100/60 text-[10px] md:text-sm mt-1">
            Reconocimiento inteligente
          </p>
        </div>

        {running && (
          <div className="flex items-center gap-2 bg-black/40 border border-cyan-400/30 px-3 py-2 rounded-2xl backdrop-blur-md">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />

            <span className="text-cyan-200 text-xs md:text-sm font-semibold">
              LIVE
            </span>
          </div>
        )}
      </div>

      {/* START */}
      {!running && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-sm z-50">
          <div className="flex flex-col items-center gap-6 px-8 text-center">
            <div className="w-28 h-28 rounded-full border-4 border-cyan-400 shadow-[0_0_60px_rgba(0,255,255,0.7)] flex items-center justify-center animate-pulse">
              <div className="w-16 h-16 rounded-full bg-cyan-400/20 border border-cyan-300" />
            </div>

            <div>
              <h1 className="text-4xl sm:text-5xl md:text-7xl font-black text-cyan-300 tracking-[0.2em]">
                FACE AI
              </h1>

              <p className="text-cyan-100/70 mt-4 text-sm md:text-lg">
                Sistema inteligente de reconocimiento facial
              </p>
            </div>

            <button
              onClick={startCamera}
              disabled={!ready}
              className="mt-4 px-8 md:px-12 py-4 rounded-3xl bg-cyan-400 hover:bg-cyan-300 active:scale-95 transition-all text-black font-black text-lg md:text-2xl shadow-[0_0_40px_rgba(0,255,255,0.7)] disabled:opacity-40"
            >
              INICIAR SISTEMA
            </button>
          </div>
        </div>
      )}
    </main>
  );
}