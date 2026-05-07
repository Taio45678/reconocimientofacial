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
              ideal: 1920,
            },

            height: {
              ideal: 1080,
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

      // DETECCIÓN OBJETOS
      const objects =
        await objectDetector.detect(video);

      objects
        .filter((obj) => obj.score > 0.60)
        .forEach((obj) => {
          const [x, y, w, h] = obj.bbox;

          ctx.strokeStyle =
            "#00ffff";

          ctx.lineWidth = 3;

          ctx.shadowColor =
            "#00ffff";

          ctx.shadowBlur = 15;

          ctx.strokeRect(x, y, w, h);

          ctx.fillStyle =
            "#00ffff";

          ctx.font =
            "bold 18px Arial";

          ctx.fillText(
            `${obj.class} ${Math.round(
              obj.score * 100
            )}%`,
            x,
            y - 10
          );
        });

      // DETECCIÓN ROSTRO
      const result =
        faceLandmarker.detectForVideo(
          video,
          now
        );

      const faces =
        result.faceLandmarks || [];

      faces.forEach((landmarks) => {
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          {
            color:
              "rgba(0,255,180,0.22)",

            lineWidth: 1,
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

        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          {
            color: "#ffcc00",

            lineWidth: 2,
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

        ctx.strokeStyle =
          "#00ff88";

        ctx.lineWidth = 3;

        ctx.shadowColor =
          "#00ff88";

        ctx.shadowBlur = 20;

        ctx.strokeRect(
          minX,
          minY,
          faceWidth,
          faceHeight
        );
      });

      animationRef.current =
        requestAnimationFrame(render);
    }

    render();
  }

  return (
    <main className="min-h-screen bg-black flex items-center justify-center overflow-hidden">
      <div className="w-full h-screen relative">
        <canvas
          ref={canvasRef}
          className="w-full h-full object-cover bg-black"
        />

        <video
          ref={videoRef}
          className="hidden"
          playsInline
          muted
        />

        {!running && (
          <div className="absolute inset-0 flex items-center justify-center bg-black">
            <button
              onClick={startCamera}
              disabled={!ready}
              className="
                px-10
                py-5
                rounded-3xl
                bg-cyan-400
                hover:bg-cyan-300
                transition-all
                text-black
                font-black
                text-2xl
                shadow-2xl
                disabled:opacity-40
              "
            >
              INICIAR IA
            </button>
          </div>
        )}
      </div>
    </main>
  );
}