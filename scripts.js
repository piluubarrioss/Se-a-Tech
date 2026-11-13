
import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ======================================
// ELEMENTOS
// ======================================

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startCameraButton = document.getElementById("startCamera");
const stopCameraButton = document.getElementById("stopCamera");
const captureSampleButton = document.getElementById("captureSample");

const cameraStatus = document.getElementById("cameraStatus");
const liveStatus = document.getElementById("liveStatus");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

const trainingLetter = document.getElementById("trainingLetter");
const selectedLetter = document.getElementById("selectedLetter");
const sampleCount = document.getElementById("sampleCount");
const trainingProgress = document.getElementById("trainingProgress");
const trainingStatus = document.getElementById("trainingStatus");
const trainingSummary = document.getElementById("trainingSummary");
const classifierStatus = document.getElementById("classifierStatus");

const prediction = document.getElementById("prediction");
const predictionMessage = document.getElementById("predictionMessage");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const addPredictionButton = document.getElementById("addPrediction");

const wordOutput = document.getElementById("wordOutput");
const alphabetGrid = document.getElementById("alphabetGrid");

const addSpaceButton = document.getElementById("addSpace");
const deleteLastButton = document.getElementById("deleteLast");
const clearWordButton = document.getElementById("clearWord");

const deleteLetterSamplesButton = document.getElementById("deleteLetterSamples");
const clearAllTrainingButton = document.getElementById("clearAllTraining");

// ======================================
// CONFIGURACIÓN
// ======================================

const LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ".split("");
const MIN_SAMPLES = 8;
const STORAGE_KEY = "senatech_training_data";

let handLandmarker = null;
let cameraStream = null;
let cameraRunning = false;
let animationFrame = null;

let lastFeatures = null;
let lastPrediction = "";
let predictionHistory = [];

let trainingData = loadTrainingData();
let word = "";

// ======================================
// MENSAJES
// ======================================

function setCameraMessage(message, type = "") {
    cameraStatus.textContent = message;
    cameraStatus.className = "message";

    if (type) {
        cameraStatus.classList.add(type);
    }
}

function setTrainingMessage(message, type = "") {
    trainingStatus.textContent = message;
    trainingStatus.className = "message";

    if (type) {
        trainingStatus.classList.add(type);
    }
}

// ======================================
// GUARDAR ENTRENAMIENTOS
// ======================================

function loadTrainingData() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        return saved ? JSON.parse(saved) : {};
    } catch (error) {
        console.error("Error al cargar entrenamientos:", error);
        return {};
    }
}

function saveTrainingData() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trainingData));
        return true;
    } catch (error) {
        console.error("Error al guardar:", error);
        setTrainingMessage("No se pudieron guardar los ejemplos.", "error");
        return false;
    }
}

// ======================================
// CARGAR MEDIAPIPE
// ======================================

async function loadHandLandmarker() {
    if (handLandmarker) return;

    setCameraMessage("Cargando sistema de reconocimiento...");

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    const options = {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    };

    try {
        handLandmarker = await HandLandmarker.createFromOptions(
            vision,
            options
        );
    } catch (gpuError) {
        console.warn("GPU no disponible. Probando CPU...", gpuError);

        options.baseOptions.delegate = "CPU";

        handLandmarker = await HandLandmarker.createFromOptions(
            vision,
            options
        );
    }

    classifierStatus.textContent = "Reconocedor preparado";
}

// ======================================
// CÁMARA
// ======================================

async function startCamera() {
    if (cameraRunning) return;

    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Abrí la página con Live Server o localhost para usar la cámara."
            );
        }

        setCameraMessage("Iniciando cámara...");

        await loadHandLandmarker();

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        cameraRunning = true;

        cameraPlaceholder.style.display = "none";
        document.querySelector(".camera-container")
            .classList.add("camera-active");

        setCameraMessage("Cámara activada correctamente.", "success");

        liveStatus.textContent = "Buscando manos";

        predictionHistory = [];
        lastPrediction = "";
        lastFeatures = null;

        clearPrediction();

        renderLoop();

    } catch (error) {
        console.error(error);

        let message = "No se pudo iniciar la cámara.";

        if (error.name === "NotAllowedError") {
            message = "Tenés que permitir el acceso a la cámara.";
        } else if (error.name === "NotFoundError") {
            message = "No se encontró una cámara conectada.";
        } else if (error.name === "NotReadableError") {
            message = "La cámara está siendo utilizada por otra aplicación.";
        } else if (error.message) {
            message = error.message;
        }

        setCameraMessage(message, "error");
        stopCameraTracks();
    }
}

function stopCameraTracks() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;

    cameraPlaceholder.style.display = "";
    document.querySelector(".camera-container")
        .classList.remove("camera-active");
}

function stopCamera() {
    cameraRunning = false;

    if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }

    stopCameraTracks();

    lastFeatures = null;
    lastPrediction = "";
    predictionHistory = [];

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    liveStatus.textContent = "Cámara apagada";

    setCameraMessage("Cámara desconectada.");

    clearPrediction();
}

function clearPrediction() {
    prediction.textContent = "?";
    predictionMessage.textContent = "Esperando una seña...";
    confidence.textContent = "0%";
    confidenceBar.style.width = "0%";
}

// ======================================
// CARACTERÍSTICAS DE LAS MANOS
// ======================================

function extractHandFeatures(landmarks) {
    const wrist = landmarks[0];

    const points = landmarks.map(point => ({
        x: point.x - wrist.x,
        y: point.y - wrist.y,
        z: point.z - wrist.z
    }));

    let maxDistance = 0;

    for (const point of points) {
        const distance = Math.sqrt(
            point.x ** 2 + point.y ** 2 + point.z ** 2
        );

        maxDistance = Math.max(maxDistance, distance);
    }

    if (maxDistance === 0) maxDistance = 1;

    const features = [];

    for (const point of points) {
        features.push(
            point.x / maxDistance,
            point.y / maxDistance,
            point.z / maxDistance
        );
    }

    return features;
}

function createFeatureVector(results) {
    const hands = results.landmarks || [];

    if (hands.length === 0) return null;

    const sortedHands = [...hands].sort((a, b) => {
        const centerA = a.reduce((sum, point) => sum + point.x, 0) / 21;
        const centerB = b.reduce((sum, point) => sum + point.x, 0) / 21;

        return centerA - centerB;
    });

    const vector = [];

    for (let i = 0; i < 2; i++) {
        if (sortedHands[i]) {
            vector.push(1);
            vector.push(...extractHandFeatures(sortedHands[i]));
        } else {
            vector.push(...Array(64).fill(0));
        }
    }

    return vector;
}

// ======================================
// DIBUJAR MANOS
// ======================================

function drawHands(results) {
    if (!video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [5, 9], [9, 10], [10, 11], [11, 12],
        [9, 13], [13, 14], [14, 15], [15, 16],
        [13, 17], [17, 18], [18, 19], [19, 20],
        [0, 17]
    ];

    for (const hand of results.landmarks || []) {
        ctx.lineWidth = 4;
        ctx.strokeStyle = "#56cfe1";
        ctx.fillStyle = "#ffffff";

        for (const [start, end] of connections) {
            ctx.beginPath();

            ctx.moveTo(
                hand[start].x * canvas.width,
                hand[start].y * canvas.height
            );

            ctx.lineTo(
                hand[end].x * canvas.width,
                hand[end].y * canvas.height
            );

            ctx.stroke();
        }

        for (const point of hand) {
            ctx.beginPath();

            ctx.arc(
                point.x * canvas.width,
                point.y * canvas.height,
                5,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }
}

// ======================================
// CLASIFICADOR
// ======================================

function calculateDistance(a, b) {
    let sum = 0;

    for (let i = 0; i < a.length; i++) {
        sum += (a[i] - b[i]) ** 2;
    }

    return Math.sqrt(sum);
}

function classify(features) {
    const samples = [];

    for (const letter of LETTERS) {
        const examples = trainingData[letter] || [];

        for (const example of examples) {
            if (example.length === features.length) {
                samples.push({
                    letter,
                    distance: calculateDistance(features, example)
                });
            }
        }
    }

    if (samples.length < MIN_SAMPLES) return null;

    samples.sort((a, b) => a.distance - b.distance);

    const nearest = samples.slice(0, 5);
    const votes = {};

    for (const sample of nearest) {
        const weight = 1 / (sample.distance + 0.001);

        votes[sample.letter] =
            (votes[sample.letter] || 0) + weight;
    }

    const ranking = Object.entries(votes)
        .sort((a, b) => b[1] - a[1]);

    const total = ranking.reduce((sum, item) => sum + item[1], 0);

    return {
        letter: ranking[0][0],
        confidence: total > 0 ? ranking[0][1] / total : 0
    };
}

// ======================================
// PREDICCIÓN ESTABLE
// ======================================

function updateStablePrediction(result) {
    predictionHistory.push(result.letter);

    if (predictionHistory.length > 10) {
        predictionHistory.shift();
    }

    const counts = {};

    for (const letter of predictionHistory) {
        counts[letter] = (counts[letter] || 0) + 1;
    }

    const winner = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])[0];

    if (!winner || winner[1] < 7) return;

    lastPrediction = winner[0];

    prediction.textContent = lastPrediction;
    predictionMessage.textContent = "Seña detectada: " + lastPrediction;

    const percentage = Math.round(result.confidence * 100);

    confidence.textContent = percentage + "%";
    confidenceBar.style.width = percentage + "%";
}

// ======================================
// RECONOCIMIENTO EN VIVO
// ======================================

function renderLoop() {
    if (!cameraRunning || !handLandmarker) return;

    try {
        if (video.readyState >= 2) {
            const results = handLandmarker.detectForVideo(
                video,
                performance.now()
            );

            drawHands(results);

            lastFeatures = createFeatureVector(results);

            if (lastFeatures) {
                liveStatus.textContent = "Mano detectada";

                const result = classify(lastFeatures);

                if (result && result.confidence >= 0.40) {
                    updateStablePrediction(result);
                } else {
                    predictionHistory = [];
                    lastPrediction = "";
                    clearPrediction();

                    predictionMessage.textContent =
                        "No se reconoce la seña con seguridad.";
                }

            } else {
                liveStatus.textContent =
                    "Mostrá tu mano frente a la cámara";

                predictionHistory = [];
                lastPrediction = "";

                clearPrediction();
            }
        }

    } catch (error) {
        console.error("Error durante el reconocimiento:", error);
    }

    animationFrame = requestAnimationFrame(renderLoop);
}

// ======================================
// ENTRENAMIENTO
// ======================================

function captureTrainingSample() {
    const letter = trainingLetter.value;

    if (!cameraRunning) {
        setTrainingMessage("Primero activá la cámara.", "error");
        return;
    }

    if (!lastFeatures) {
        setTrainingMessage("No se detectó ninguna mano.", "error");
        return;
    }

    if (!trainingData[letter]) {
        trainingData[letter] = [];
    }

    trainingData[letter].push([...lastFeatures]);

    if (!saveTrainingData()) {
        trainingData[letter].pop();
        return;
    }

    setTrainingMessage(
        "¡Ejemplo guardado para la letra " + letter + "!",
        "success"
    );

    updateTrainingInterface();
}

function deleteCurrentLetterSamples() {
    const letter = trainingLetter.value;

    delete trainingData[letter];

    saveTrainingData();

    setTrainingMessage(
        "Se eliminaron los ejemplos de " + letter + "."
    );

    updateTrainingInterface();
}

function clearAllTraining() {
    const confirmed = confirm(
        "¿Seguro que querés borrar todos los ejemplos guardados?"
    );

    if (!confirmed) return;

    trainingData = {};

    saveTrainingData();

    setTrainingMessage("Se eliminaron todos los entrenamientos.");

    updateTrainingInterface();
}

// ======================================
// ACTUALIZAR ENTRENAMIENTO
// ======================================

function updateTrainingInterface() {
    const letter = trainingLetter.value || "A";
    const examples = trainingData[letter] || [];
    const count = examples.length;

    sampleCount.textContent = count;
    selectedLetter.textContent = letter;

    trainingProgress.style.width =
        Math.min((count / MIN_SAMPLES) * 100, 100) + "%";

    const total = LETTERS.reduce(
        (sum, item) => sum + (trainingData[item]?.length || 0),
        0
    );

    trainingSummary.textContent =
        total + " ejemplos guardados en total.";

    classifierStatus.textContent = total >= MIN_SAMPLES
        ? "Entrenamiento disponible."
        : "Faltan ejemplos de entrenamiento.";

    if (count >= MIN_SAMPLES) {
        setTrainingMessage(
            "La letra " + letter + " tiene suficientes ejemplos.",
            "success"
        );
    } else {
        setTrainingMessage(
            "Necesitás al menos " + MIN_SAMPLES +
            " ejemplos. Tenés " + count + "."
        );
    }
}

// ======================================
// ABECEDARIO
// ======================================

function createLetterOptions() {
    trainingLetter.innerHTML = "";

    for (const letter of LETTERS) {
        const option = document.createElement("option");

        option.value = letter;
        option.textContent = letter;

        trainingLetter.appendChild(option);
    }
}

function createAlphabet() {
    alphabetGrid.innerHTML = "";

    for (const letter of LETTERS) {
        const button = document.createElement("button");

        button.type = "button";
        button.textContent = letter;
        button.className = "alphabet-letter";

        button.addEventListener("click", () => {
            trainingLetter.value = letter;
            updateTrainingInterface();
        });

        alphabetGrid.appendChild(button);
    }
}

// ======================================
// ARMAR PALABRAS
// ======================================

function updateWord() {
    wordOutput.textContent = word || "Tu palabra aparecerá acá...";
}

function addPredictionToWord() {
    if (!lastPrediction) {
        setTrainingMessage(
            "Primero esperá a que se reconozca una letra.",
            "error"
        );

        return;
    }

    word += lastPrediction;
    updateWord();
}

function addSpace() {
    if (word && !word.endsWith(" ")) {
        word += " ";
        updateWord();
    }
}

function deleteLastCharacter() {
    word = word.slice(0, -1);
    updateWord();
}

function clearWord() {
    word = "";
    updateWord();
}

// ======================================
// EVENTOS
// ======================================

startCameraButton.addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", stopCamera);

captureSampleButton.addEventListener(
    "click",
    captureTrainingSample
);

trainingLetter.addEventListener(
    "change",
    updateTrainingInterface
);

addPredictionButton.addEventListener(
    "click",
    addPredictionToWord
);

addSpaceButton.addEventListener("click", addSpace);
deleteLastButton.addEventListener("click", deleteLastCharacter);
clearWordButton.addEventListener("click", clearWord);

deleteLetterSamplesButton.addEventListener(
    "click",
    deleteCurrentLetterSamples
);

clearAllTrainingButton.addEventListener(
    "click",
    clearAllTraining
);

// ======================================
// INICIO
// ======================================

createLetterOptions();
createAlphabet();
updateTrainingInterface();
updateWord();

window.addEventListener("beforeunload", () => {
    stopCameraTracks();

    if (handLandmarker) {
        handLandmarker.close();
    }
});




