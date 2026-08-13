"use strict";

const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;
const WORD_FRAME_TOTAL = 50;
const WORD_SAMPLE_INTERVAL_MS = 40; // Maximum 25 samples per second.
const HOLISTIC_INTERVAL_MS = 33;
const ALPHABET_SEND_INTERVAL_MS = 100;
const LETTER_ACCEPT_COOLDOWN_MS = 1_000;
const REQUEST_TIMEOUT_MS = 8_000;
const WORD_RESULT_TIMEOUT_MS = 20_000;
const TRANSLATION_REQUEST_TIMEOUT_MS = 120_000;
const MAX_RECORDING_MS = 30_000;
const MAX_SUGGESTIONS = 8;
const SUGGESTION_DEBOUNCE_MS = 160;
const CONTEXT_SUGGESTION_ENDPOINT = "/api/word-suggestions";
const MAX_RECONNECT_ATTEMPTS = 8;
const MAX_RECONNECT_DELAY_MS = 15_000;
const MAX_WS_BUFFERED_BYTES = 512_000;
const MEDIAPIPE_HOLISTIC_BASE = "/static/assets/mediapipe";
const MEDIAPIPE_DRAWING_URL = `${MEDIAPIPE_HOLISTIC_BASE}/drawing_utils.js`;
const MEDIAPIPE_HOLISTIC_URL = `${MEDIAPIPE_HOLISTIC_BASE}/holistic.js`;
const ALPHABET_HAND_BONES = Object.freeze([
    [0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0],
    [0, 5, 1], [5, 6, 1], [6, 7, 1], [7, 8, 1],
    [0, 9, 2], [9, 10, 2], [10, 11, 2], [11, 12, 2],
    [0, 13, 3], [13, 14, 3], [14, 15, 3], [15, 16, 3],
    [0, 17, 4], [17, 18, 4], [18, 19, 4], [19, 20, 4],
    [5, 9, 1], [9, 13, 2], [13, 17, 3],
]);
const ALPHABET_HAND_COLORS = Object.freeze(["#b28dff", "#65e6ff", "#58efad", "#ffd26d", "#ff91b3"]);
let wordTrackingLibrariesPromise = null;
let wordTrackerInitializationPromise = null;

// Kept in the client so autocomplete remains useful if the optional vocabulary
// request is temporarily unavailable. The API result replaces this list when
// it arrives.
const FALLBACK_VOCABULARY = Object.freeze([
    "again", "angry", "baby", "bad", "big", "blue", "brother", "can", "car", "child",
    "close", "cold", "color", "come", "different", "drink", "eat", "family", "father", "feel",
    "finish", "food", "friend", "give", "go", "good", "green", "happy", "hear", "hello",
    "help", "home", "hot", "how", "hungry", "know", "learn", "like", "love", "man",
    "money", "more", "morning", "mother", "name", "need", "new", "night", "no", "old",
    "open", "pain", "play", "please", "practice", "read", "red", "right", "run", "sad",
    "same", "say", "school", "see", "sick", "sit", "sister", "sleep", "small", "sorry",
    "stand", "stop", "take", "teach", "tell", "thank you", "think", "thirsty", "time", "tired",
    "today", "tomorrow", "understand", "wait", "wake", "walk", "want", "water", "week", "what",
    "when", "where", "who", "why", "woman", "work", "write", "wrong", "yes", "yesterday",
]);

const state = {
    disposed: false,
    websocket: null,
    wsPhase: "idle",
    reconnectTimer: null,
    reconnectAttempt: 0,
    healthReachable: false,
    healthStatus: "checking",
    healthLoaded: false,
    capabilities: {
        alphabet: { ready: false, error: "", metadata: {} },
        words: { ready: false, error: "", metadata: {}, experimental: true },
        speechToSign: { ready: false, textReady: false, voiceReady: false, status: "checking", error: "", metadata: {} },
    },
    mode: null,
    cameraStream: null,
    cameraRunning: false,
    cameraStarting: false,
    holistic: null,
    holisticBusy: false,
    wordTracker: {
        status: "idle",
        error: "",
    },
    cameraAnimationFrame: null,
    lastHolisticRequestAt: 0,
    fpsFrames: 0,
    fpsTimer: null,
    requestCounter: 0,
    requestSentAt: new Map(),
    alphabet: {
        inFlight: null,
        timeout: null,
        encoding: false,
        lastSentAt: 0,
        armed: true,
        lastAcceptedAt: Number.NEGATIVE_INFINITY,
        lastAcceptedLabel: "",
    },
    words: {
        active: false,
        waiting: false,
        needsReset: false,
        captured: 0,
        requestId: null,
        lastSampleAt: 0,
        timeout: null,
    },
    activePage: "sign-input",
    keyboard: {
        spelling: "",
        sentence: "",
        currentLetter: "",
        pendingLetter: "",
        pendingConfidence: 0,
        pendingLetterAcceptable: false,
        pendingLetterCommitted: true,
        suggestions: [],
        selectedSuggestionIndex: 0,
        suggestionStatus: "idle",
        suggestionSource: "local",
        suggestionQueryKey: "",
        suggestionRequestNumber: 0,
        suggestionAbortController: null,
        suggestionTimer: null,
        vocabulary: [...FALLBACK_VOCABULARY],
        vocabularySource: "fallback",
        soundEnabled: true,
        audioContext: null,
        cooldownTimer: null,
    },
    translation: {
        busy: false,
        clips: [],
        activeIndex: 0,
        mediaRecorder: null,
        micStream: null,
        chunks: [],
        shouldTranslateRecording: false,
        recordingStartedAt: 0,
        recordingTimer: null,
        maxRecordingTimer: null,
        recognition: null,
        recognitionActive: false,
        recognitionStopRequested: false,
        recognitionFinalTranscript: "",
        recognitionInterimTranscript: "",
        recognitionError: "",
        recognitionBaseText: "",
        preferRecordedAudio: true,
    },
    history: [],
};

const DOM = {};
const alphabetFrameCanvas = document.createElement("canvas");
alphabetFrameCanvas.width = FRAME_WIDTH;
alphabetFrameCanvas.height = FRAME_HEIGHT;

function cacheDom() {
    const ids = [
        "backend-status-indicator", "backend-status", "camera-status-indicator", "camera-status",
        "alphabet-status-indicator", "alphabet-status", "words-status-indicator", "words-status",
        "retry-backend", "system-message", "video-stage", "webcam", "overlay", "camera-placeholder",
        "fps-display", "start-camera", "stop-camera", "mode-alphabet", "mode-words",
        "words-experimental-badge", "mode-guidance", "words-capture-panel", "capture-guidance",
        "start-word-capture", "capture-progress", "capture-progress-fill", "capture-progress-text",
        "prediction-state", "prediction-text", "prediction-detail", "confidence-progress",
        "confidence-fill", "confidence-text", "latency-display", "server-latency-display",
        "stream-status", "top-predictions-list", "history-list", "clear-history", "reset-sign",
        "nav-sign-input", "nav-text-to-sign", "page-sign-input", "page-text-to-sign",
        "sound-toggle", "sound-toggle-label", "current-letter", "current-spelling", "input-readiness",
        "word-suggestions", "suggestion-help", "keyboard-backspace", "keyboard-clear-letters",
        "keyboard-space", "keyboard-confirm", "keyboard-clear-sentence", "sentence-output",
        "text-sign-form", "translation-input", "translation-char-count", "translate-text",
        "clear-translation-input", "start-recording", "start-recording-label", "stop-recording",
        "stop-recording-label", "recording-time", "voice-input-mode", "voice-live-transcript",
        "voice-privacy-note",
        "voice-status", "translation-state", "translation-empty", "translation-summary",
        "translation-transcript", "normalized-tokens", "translation-note", "translation-message",
        "sign-results-card", "sign-count", "sign-sequence",
        "sequence-position", "previous-sign", "sequence-action", "next-sign",
        "text-service-indicator", "text-service-status", "voice-service-indicator",
        "voice-service-status", "translation-service-message",
    ];

    for (const id of ids) {
        DOM[toCamelCase(id)] = document.getElementById(id);
    }
}

function toCamelCase(value) {
    return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
}

function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function safeText(value, fallback = "") {
    return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function normalizeDetectedLetter(value) {
    const normalized = safeText(value).toUpperCase();
    return /^[A-Z]$/.test(normalized) ? normalized : "";
}

function normalizeOverlayLandmarks(value) {
    if (!Array.isArray(value) || value.length !== 21) return [];
    const landmarks = value.map((point) => {
        const x = finiteNumber(Array.isArray(point) ? point[0] : point?.x);
        const y = finiteNumber(Array.isArray(point) ? point[1] : point?.y);
        if (x === null || y === null) return null;
        return { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
    });
    return landmarks.every(Boolean) ? landmarks : [];
}

/**
 * Pure acceptance state machine used by live recognition and deterministic
 * browser tests. A held pose cannot repeat: the same letter requires a no-hand
 * re-arm as well as the global cooldown. A different accepted letter can re-arm
 * the stream naturally, but never bypasses that cooldown.
 */
function getLetterAcceptanceTransition({
    label,
    armed,
    lastAcceptedLabel,
    lastAcceptedAt,
    now,
    cooldownMs = LETTER_ACCEPT_COOLDOWN_MS,
}) {
    const letter = normalizeDetectedLetter(label);
    const currentTime = finiteNumber(now) ?? 0;
    const previousTime = finiteNumber(lastAcceptedAt) ?? Number.NEGATIVE_INFINITY;
    const elapsed = currentTime - previousTime;
    const remainingMs = Math.max(0, cooldownMs - elapsed);

    if (!letter) {
        return { accepted: false, reason: "invalid", armed: Boolean(armed), lastAcceptedLabel, lastAcceptedAt, remainingMs: 0 };
    }
    if (remainingMs > 0) {
        return { accepted: false, reason: "cooldown", armed: Boolean(armed), lastAcceptedLabel, lastAcceptedAt, remainingMs };
    }

    const changedLetter = Boolean(lastAcceptedLabel) && letter !== lastAcceptedLabel;
    if (!armed && !changedLetter) {
        return { accepted: false, reason: "held", armed: false, lastAcceptedLabel, lastAcceptedAt, remainingMs: 0 };
    }

    return {
        accepted: true,
        reason: "accepted",
        armed: false,
        lastAcceptedLabel: letter,
        lastAcceptedAt: currentTime,
        remainingMs: cooldownMs,
    };
}

function buildSuggestions(prefix, vocabulary, limit = MAX_SUGGESTIONS) {
    const needle = safeText(prefix).toLowerCase().replace(/[^a-z]/g, "");
    if (!needle) return [];
    const unique = new Set();
    for (const entry of Array.isArray(vocabulary) ? vocabulary : []) {
        const word = safeText(entry).toLowerCase().replace(/\s+/g, " ");
        if (word.startsWith(needle)) unique.add(word);
    }
    return [...unique]
        .sort((first, second) => {
            const firstExact = first === needle ? 0 : 1;
            const secondExact = second === needle ? 0 : 1;
            return firstExact - secondExact || first.length - second.length || first.localeCompare(second);
        })
        .slice(0, Math.max(0, Math.floor(limit)));
}

function sentenceContextWords(sentence) {
    return safeText(sentence)
        .toLowerCase()
        .match(/[a-z]+(?:'[a-z]+)?/g) || [];
}

function suggestionItemsFromPayload(payload) {
    const root = payload?.data && typeof payload.data === "object" ? payload.data : payload;
    const candidates = [
        root?.suggestions,
        root?.predictions,
        root?.next_words,
        root?.words,
        root?.items,
        Array.isArray(root) ? root : null,
    ];
    for (const candidate of candidates) {
        if (Array.isArray(candidate)) return candidate;
        if (Array.isArray(candidate?.items)) return candidate.items;
    }
    return [];
}

function normalizeContextSuggestions(payload, prefix = "", limit = MAX_SUGGESTIONS) {
    const needle = safeText(prefix).toLowerCase().replace(/[^a-z]/g, "");
    const seen = new Set();
    return suggestionItemsFromPayload(payload)
        .map((item, index) => {
            const word = safeText(
                typeof item === "string" ? item : item?.word,
                safeText(item?.text, safeText(item?.token, safeText(item?.label))),
            ).toLowerCase().replace(/\s+/g, " ");
            const score = finiteNumber(item?.score ?? item?.probability ?? item?.confidence);
            return { word, score, index };
        })
        .filter((item) => {
            if (!item.word || (needle && !item.word.startsWith(needle)) || seen.has(item.word)) return false;
            seen.add(item.word);
            return true;
        })
        .sort((first, second) => {
            if (first.score !== null && second.score !== null && first.score !== second.score) {
                return second.score - first.score;
            }
            return first.index - second.index;
        })
        .slice(0, Math.max(0, Math.floor(limit)))
        .map((item) => item.word);
}

function commitWordTransition({ sentence = "", spelling = "" }, selectedWord) {
    const word = safeText(selectedWord, safeText(spelling)).replace(/\s+/g, " ");
    if (!word) return { sentence, spelling };
    const existing = sentence.trim();
    const displayWord = existing ? word.toLowerCase() : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    return {
        sentence: `${existing ? `${existing} ` : ""}${displayWord} `,
        spelling: "",
    };
}

window.SignBridgeTest = Object.freeze({
    getLetterAcceptanceTransition,
    buildSuggestions,
    sentenceContextWords,
    normalizeContextSuggestions,
    commitWordTransition,
    normalizeOverlayLandmarks,
});

function nextRequestId(prefix) {
    state.requestCounter += 1;
    return `${prefix}-${Date.now()}-${state.requestCounter}`;
}

function isWebSocketOpen() {
    return state.websocket && state.websocket.readyState === WebSocket.OPEN;
}

function setIndicator(element, status) {
    if (element) element.dataset.state = status;
}

function setSystemMessage(message, status = "info") {
    DOM.systemMessage.textContent = message;
    DOM.systemMessage.dataset.state = status;
}

function setCameraStatus(message, status) {
    DOM.cameraStatus.textContent = message;
    setIndicator(DOM.cameraStatusIndicator, status);
}

function renderBackendStatus() {
    const streamConnected = isWebSocketOpen();
    DOM.streamStatus.textContent = streamConnected ? "Online" : "Offline";

    if (streamConnected && state.healthReachable) {
        const degraded = state.healthStatus !== "ready" && state.healthStatus !== "healthy";
        DOM.backendStatus.textContent = degraded ? "Connected; service degraded" : "API and stream connected";
        setIndicator(DOM.backendStatusIndicator, degraded ? "warning" : "online");
        return;
    }

    if (streamConnected) {
        DOM.backendStatus.textContent = "Stream connected; readiness unknown";
        setIndicator(DOM.backendStatusIndicator, "warning");
        return;
    }

    if (state.healthReachable) {
        DOM.backendStatus.textContent = state.wsPhase === "stopped" ? "API reachable; stream retry stopped" : "API reachable; stream connecting";
        setIndicator(DOM.backendStatusIndicator, "warning");
        return;
    }

    if (state.wsPhase === "stopped") {
        DOM.backendStatus.textContent = "Unavailable; use Retry connection";
        setIndicator(DOM.backendStatusIndicator, "offline");
    } else if (state.wsPhase === "retrying") {
        DOM.backendStatus.textContent = `Reconnecting (${state.reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS})`;
        setIndicator(DOM.backendStatusIndicator, "checking");
    } else {
        DOM.backendStatus.textContent = "Connecting";
        setIndicator(DOM.backendStatusIndicator, "checking");
    }
}

function capabilityError(capability) {
    if (!capability) return "";
    if (typeof capability.error === "string") return capability.error;
    if (capability.error && typeof capability.error.message === "string") return capability.error.message;
    if (typeof capability.reason === "string") return capability.reason;
    return "";
}

function normalizeCapability(payload, name, legacyField) {
    const raw = payload?.capabilities?.[name];
    let ready = false;
    let metadata = {};

    if (typeof raw === "boolean") {
        ready = raw;
    } else if (raw && typeof raw === "object") {
        ready = raw.ready === true;
        metadata = raw.metadata && typeof raw.metadata === "object" ? raw.metadata : {};
    } else if (typeof payload?.[legacyField] === "boolean") {
        ready = payload[legacyField];
    }

    return {
        ready,
        error: capabilityError(raw),
        metadata,
        experimental: name === "words",
    };
}

function normalizeSpeechCapability(payload) {
    const raw = payload?.capabilities?.speech_to_sign;
    if (!raw || typeof raw !== "object") {
        const legacyReady = payload?.modes?.speech_to_sign === true;
        return {
            ready: legacyReady,
            textReady: legacyReady,
            voiceReady: false,
            status: legacyReady ? "ready" : "unavailable",
            error: "",
            metadata: {},
        };
    }

    const textReady = raw.text_ready === true || raw.ready === true;
    const voiceReady = raw.voice_ready === true;
    return {
        ready: textReady,
        textReady,
        voiceReady,
        status: safeText(raw.status, textReady ? "ready" : "unavailable").toLowerCase(),
        error: capabilityError(raw),
        metadata: raw.metadata && typeof raw.metadata === "object" ? raw.metadata : {},
    };
}

function renderTranslationCapabilities() {
    const capability = state.capabilities.speechToSign;
    const checked = state.healthLoaded;
    const browserSpeechReady = Boolean(speechRecognitionClass() && capability.textReady);

    DOM.textServiceStatus.textContent = capability.textReady
        ? "Ready"
        : checked ? "Unavailable" : "Checking";
    DOM.voiceServiceStatus.textContent = capability.voiceReady
        ? "Local Whisper ready"
        : browserSpeechReady
            ? "Browser speech available"
        : checked ? "Unavailable" : "Checking";
    setIndicator(DOM.textServiceIndicator, capability.textReady ? "ready" : checked ? "offline" : "checking");
    setIndicator(DOM.voiceServiceIndicator, (browserSpeechReady || capability.voiceReady) ? "ready" : checked ? "offline" : "checking");

    if (!checked) {
        DOM.translationServiceMessage.textContent = "Checking text and voice services…";
        DOM.translationServiceMessage.dataset.state = "checking";
    } else if (capability.textReady && (capability.voiceReady || browserSpeechReady)) {
        DOM.translationServiceMessage.textContent = "Text and voice translation are ready.";
        DOM.translationServiceMessage.dataset.state = "ready";
    } else if (capability.textReady) {
        DOM.translationServiceMessage.textContent = "Text translation is ready. Voice is unavailable on this server; type your message instead.";
        DOM.translationServiceMessage.dataset.state = "warning";
    } else {
        DOM.translationServiceMessage.textContent = capability.error
            ? `Translation is unavailable: ${capability.error}`
            : "Text and voice translation are unavailable on this server.";
        DOM.translationServiceMessage.dataset.state = "error";
    }
    updateTranslationControls();
}

function renderCapabilities() {
    const alphabet = state.capabilities.alphabet;
    const words = state.capabilities.words;
    const wordTracker = state.wordTracker;

    DOM.alphabetStatus.textContent = alphabet.ready
        ? "Ready"
        : state.healthLoaded
            ? (alphabet.error ? `Unavailable: ${alphabet.error}` : "Unavailable")
            : "Checking readiness";
    setIndicator(DOM.alphabetStatusIndicator, alphabet.ready ? "ready" : state.healthLoaded ? "offline" : "checking");

    let wordsStatus = "Checking backend readiness";
    let wordsIndicator = "checking";
    if (state.healthLoaded && !words.ready) {
        wordsStatus = words.error ? `Unavailable: ${words.error}` : "Unavailable";
        wordsIndicator = "offline";
    } else if (words.ready && wordTracker.status === "ready") {
        wordsStatus = "Ready (experimental)";
        wordsIndicator = "warning";
    } else if (words.ready && wordTracker.status === "loading") {
        wordsStatus = "Initializing local browser tracker";
    } else if (words.ready && wordTracker.status === "error") {
        wordsStatus = `Tracker unavailable: ${wordTracker.error}`;
        wordsIndicator = "offline";
    } else if (words.ready) {
        wordsStatus = "Local browser tracker loads when selected";
    }
    DOM.wordsStatus.textContent = wordsStatus;
    setIndicator(DOM.wordsStatusIndicator, wordsIndicator);

    DOM.wordsExperimentalBadge.classList.remove("is-hidden");
    DOM.modeAlphabet.disabled = !alphabet.ready;
    DOM.modeWords.disabled = !words.ready;

    if (state.mode === "alphabet" && !alphabet.ready) state.mode = null;
    if (state.mode === "words" && !words.ready) state.mode = null;

    if (!state.mode) {
        if (alphabet.ready) {
            setMode("alphabet", false);
        } else if (words.ready) {
            setMode("words", false);
        } else {
            renderMode();
            clearLivePrediction("No model available", "Recognition is disabled until a model reports ready.");
        }
    } else {
        renderMode();
    }

    updateControls();
    renderTranslationCapabilities();
}

async function checkHealth() {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5_000);

    try {
        const response = await fetch("/health", {
            cache: "no-store",
            headers: { Accept: "application/json" },
            signal: controller.signal,
        });
        const payload = await response.json();
        if (!payload || typeof payload !== "object") throw new Error("Invalid health response");

        state.healthReachable = true;
        state.healthLoaded = true;
        state.healthStatus = safeText(payload.status, response.ok ? "ready" : "degraded").toLowerCase();
        state.capabilities.alphabet = normalizeCapability(payload, "alphabet", "alphabet_model");
        state.capabilities.words = normalizeCapability(payload, "words", "word_model");
        state.capabilities.speechToSign = normalizeSpeechCapability(payload);
        renderCapabilities();

        const readyCount = Number(state.capabilities.alphabet.ready) + Number(state.capabilities.words.ready);
        if (readyCount === 0) {
            setSystemMessage("The backend is reachable, but no recognition model is ready.", "error");
        } else if (!state.capabilities.alphabet.ready || !state.capabilities.words.ready) {
            setSystemMessage("The backend is running in degraded mode. Unavailable recognition modes are disabled.", "warning");
        } else {
            setSystemMessage("Recognition models are ready. Word mode initializes its local browser tracker when selected.");
        }
    } catch (error) {
        state.healthReachable = false;
        state.healthLoaded = true;
        state.healthStatus = "not_ready";
        state.capabilities.alphabet = { ready: false, error: "Health check failed", metadata: {} };
        state.capabilities.words = { ready: false, error: "Health check failed", metadata: {}, experimental: true };
        state.capabilities.speechToSign = {
            ready: false,
            textReady: false,
            voiceReady: false,
            status: "unavailable",
            error: "Health check failed",
            metadata: {},
        };
        renderCapabilities();
        setSystemMessage("The backend health check failed. Start the server, then retry the connection.", "error");
    } finally {
        window.clearTimeout(timeout);
        renderBackendStatus();
    }
}

function websocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/sign-to-text`;
}

function connectWebSocket() {
    if (state.disposed) return;
    if (state.websocket && (state.websocket.readyState === WebSocket.OPEN || state.websocket.readyState === WebSocket.CONNECTING)) return;

    if (state.reconnectTimer) {
        window.clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }

    state.wsPhase = "connecting";
    renderBackendStatus();

    let socket;
    try {
        socket = new WebSocket(websocketUrl());
    } catch (error) {
        scheduleReconnect();
        return;
    }

    state.websocket = socket;

    socket.addEventListener("open", () => {
        if (state.disposed || socket !== state.websocket) return;
        state.wsPhase = "connected";
        state.reconnectAttempt = 0;
        renderBackendStatus();
        updateControls();
        renderKeyboard();
    });

    socket.addEventListener("message", (event) => {
        if (socket !== state.websocket) return;
        handleWebSocketMessage(event.data);
    });

    socket.addEventListener("error", () => {
        if (socket === state.websocket) state.wsPhase = "retrying";
    });

    socket.addEventListener("close", () => {
        if (socket !== state.websocket) return;
        state.websocket = null;
        clearAlphabetRequest();
        if (state.words.active || state.words.waiting) {
            abortWordCapture("The stream disconnected. Reset the sign after reconnection.");
        }
        renderBackendStatus();
        updateControls();
        renderKeyboard();
        if (!state.disposed) scheduleReconnect();
    });
}

function scheduleReconnect() {
    if (state.disposed || state.reconnectTimer) return;
    if (state.reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
        state.wsPhase = "stopped";
        renderBackendStatus();
        setSystemMessage("Automatic stream reconnection stopped. Use Retry connection when the backend is available.", "warning");
        return;
    }

    const baseDelay = Math.min(1_000 * (2 ** state.reconnectAttempt), MAX_RECONNECT_DELAY_MS);
    const jitter = Math.round(baseDelay * Math.random() * 0.2);
    const delay = Math.min(baseDelay + jitter, MAX_RECONNECT_DELAY_MS);
    state.reconnectAttempt += 1;
    state.wsPhase = "retrying";
    renderBackendStatus();
    state.reconnectTimer = window.setTimeout(() => {
        state.reconnectTimer = null;
        connectWebSocket();
    }, delay);
}

function sendWebSocket(payload, trackTiming = false) {
    if (!isWebSocketOpen()) return false;
    if (state.websocket.bufferedAmount > MAX_WS_BUFFERED_BYTES) return false;

    try {
        state.websocket.send(JSON.stringify(payload));
        if (trackTiming && payload.request_id) {
            state.requestSentAt.set(payload.request_id, finiteNumber(payload.sent_at) ?? Date.now());
            while (state.requestSentAt.size > 100) {
                state.requestSentAt.delete(state.requestSentAt.keys().next().value);
            }
        }
        return true;
    } catch (error) {
        return false;
    }
}

function parseError(payload) {
    if (typeof payload?.error === "string") return { code: safeText(payload.code, "error"), message: payload.error };
    if (payload?.error && typeof payload.error === "object") {
        return {
            code: safeText(payload.error.code, safeText(payload.code, "error")),
            message: safeText(payload.error.message, "Recognition request failed"),
        };
    }
    return {
        code: safeText(payload?.code, "error"),
        message: safeText(payload?.message, "Recognition request failed"),
    };
}

function handleWebSocketMessage(rawMessage) {
    let payload;
    try {
        payload = JSON.parse(rawMessage);
    } catch (error) {
        setSystemMessage("The backend sent an invalid response.", "error");
        return;
    }

    if (!payload || typeof payload !== "object") return;
    updateLatency(payload);

    if (payload.type === "progress") {
        handleProgress(payload);
        return;
    }

    if (payload.type === "no_hand" || payload.reason === "no_hand" || payload.code === "no_hand") {
        handleNoHand(payload);
        return;
    }

    if (payload.type === "error" || payload.error) {
        handleRecognitionError(payload);
        return;
    }

    if (payload.type === "status") {
        const message = safeText(payload.message);
        if (message) setSystemMessage(message);
        return;
    }

    const hasPrediction = payload.type === "prediction"
        || payload.prediction !== undefined
        || payload.result !== undefined
        || payload.sign !== undefined
        || payload.label !== undefined
        || payload.candidate_label !== undefined;
    if (hasPrediction) handlePrediction(payload);
}

function responseLabel(payload, accepted) {
    const prediction = payload.prediction;
    const objectPrediction = prediction && typeof prediction === "object" ? prediction : null;
    const directPrediction = typeof prediction === "string" ? prediction : "";
    const acceptedLabel = directPrediction
        || safeText(objectPrediction?.label)
        || safeText(payload.label)
        || safeText(payload.sign)
        || safeText(payload.result);
    const candidate = safeText(payload.candidate_label)
        || safeText(objectPrediction?.candidate_label)
        || acceptedLabel;
    return accepted ? acceptedLabel : candidate;
}

function handlePrediction(payload) {
    const mode = payload.mode === "words" || payload.mode === "alphabet" ? payload.mode : state.mode;
    const requestId = safeText(payload.request_id);

    if (mode === "alphabet") {
        clearAlphabetRequest(requestId);
        state.fpsFrames += 1;
    }
    if (mode === "words") finishWordRequest();

    const accepted = payload.accepted === true;
    const label = responseLabel(payload, accepted);
    const confidence = clamp(finiteNumber(payload.confidence) ?? 0, 0, 1);
    const topPredictions = Array.isArray(payload.top_predictions)
        ? payload.top_predictions
        : Array.isArray(payload.top5)
            ? payload.top5
            : [];

    renderTopPredictions(topPredictions);
    setConfidence(confidence);

    if (mode === "alphabet") {
        drawAlphabetHandOverlay(payload.landmarks, label, confidence, accepted);
        const detectedLetter = normalizeDetectedLetter(label);
        state.keyboard.pendingLetter = detectedLetter;
        state.keyboard.pendingConfidence = confidence;
        state.keyboard.pendingLetterAcceptable = Boolean(detectedLetter && accepted);
        state.keyboard.pendingLetterCommitted = false;
    }

    if (!label) {
        clearLivePrediction("No accepted sign", reasonText(payload.reason, "The model did not accept this frame."));
        return;
    }

    DOM.predictionText.className = "prediction-text";
    DOM.predictionText.textContent = label;
    DOM.predictionState.dataset.state = accepted ? "accepted" : "uncertain";
    DOM.predictionState.textContent = accepted ? "Accepted" : "Uncertain";

    if (!accepted) {
        DOM.predictionText.classList.add("uncertain");
        DOM.predictionDetail.textContent = `${reasonText(payload.reason, "Candidate below the acceptance threshold.")} It was not added to history.`;
        return;
    }

    if (mode === "alphabet") {
        const transition = tryAcceptAlphabetLetter(label, confidence);
        state.keyboard.pendingLetterCommitted = transition.accepted;
        if (transition.accepted) {
            DOM.predictionDetail.textContent = "Letter added. Pause briefly, then lower your hand or present a different letter.";
        } else if (transition.reason === "cooldown") {
            DOM.predictionDetail.textContent = `Letter recognized; waiting ${(transition.remainingMs / 1_000).toFixed(1)}s before another can be added.`;
        } else if (transition.reason === "held") {
            DOM.predictionDetail.textContent = "Held sign recognized but not repeated. Lower your hand before signing this letter again.";
        } else {
            DOM.predictionDetail.textContent = "The accepted result was not a single alphabet letter.";
        }
    } else {
        addToHistory(label, confidence, mode || "words");
        DOM.predictionDetail.textContent = "Isolated word accepted. Reset the current sign before another capture.";
    }
}

function reasonText(reason, fallback) {
    if (typeof reason !== "string" || !reason.trim()) return fallback;
    return reason.replaceAll("_", " ");
}

function handleNoHand(payload) {
    const mode = payload.mode === "words" || payload.mode === "alphabet" ? payload.mode : state.mode;
    if (mode === "alphabet") {
        clearAlphabetRequest(safeText(payload.request_id));
        clearOverlay();
        state.alphabet.armed = true;
        state.keyboard.pendingLetter = "";
        state.keyboard.pendingLetterAcceptable = false;
        state.keyboard.pendingLetterCommitted = true;
        state.fpsFrames += 1;
        renderKeyboard();
    }
    if (mode === "words" && (state.words.active || state.words.waiting)) {
        abortWordCapture("No usable hand sequence was detected. Reset and try again.");
    }
    clearLivePrediction("No hand detected", "Place your signing hand and upper body inside the frame.");
}

function handleRecognitionError(payload) {
    const error = parseError(payload);
    const requestId = safeText(payload.request_id);
    clearAlphabetRequest(requestId);

    if (state.words.requestId && (!requestId || requestId === state.words.requestId)) {
        abortWordCapture(`${error.message}. Reset the current sign before trying again.`);
    }

    DOM.predictionText.className = "prediction-text error";
    DOM.predictionText.textContent = "Recognition error";
    DOM.predictionDetail.textContent = error.message;
    DOM.predictionState.dataset.state = "error";
    DOM.predictionState.textContent = "Error";
    setConfidence(0);
    renderTopPredictions([]);
    setSystemMessage(error.message, "error");

    if (["model_unavailable", "model_not_ready", "service_unavailable"].includes(error.code)) {
        checkHealth();
    }
}

function handleProgress(payload) {
    if (payload.mode !== "words") return;
    const requestId = safeText(payload.request_id);
    if (state.words.requestId && requestId && requestId !== state.words.requestId) return;

    const captured = clamp(Math.round(finiteNumber(payload.captured ?? payload.current) ?? state.words.captured), 0, WORD_FRAME_TOTAL);
    const total = clamp(Math.round(finiteNumber(payload.total) ?? WORD_FRAME_TOTAL), 1, WORD_FRAME_TOTAL);
    updateWordProgress(captured, total);

    if (payload.awaiting_reset === true) {
        state.words.active = false;
        state.words.waiting = false;
        state.words.needsReset = true;
        DOM.captureGuidance.textContent = "Capture complete. Reset the current sign before another capture.";
        updateControls();
    }
}

function updateLatency(payload) {
    const requestId = safeText(payload.request_id);
    const sentAt = requestId ? state.requestSentAt.get(requestId) : null;
    if (sentAt) {
        DOM.latencyDisplay.textContent = `${Math.max(0, Math.round(Date.now() - sentAt))} ms`;
    }

    // Connection/status envelopes carry a structural 0 ms value; showing it as
    // inference latency before any inference has happened is misleading.
    if (["prediction", "no_hand", "error"].includes(payload.type)) {
        const latency = payload.latency && typeof payload.latency === "object" ? payload.latency : {};
        const server = latency.server && typeof latency.server === "object" ? latency.server : {};
        const serverMs = firstFinite([
            server.total_ms,
            latency.server_total_ms,
            latency.total_ms,
            latency.inference_ms,
            payload.server_latency_ms,
            payload.latency_ms,
        ]);
        DOM.serverLatencyDisplay.textContent = serverMs === null ? "--" : `${Math.round(serverMs)} ms`;
    }

    if (requestId && ["prediction", "no_hand", "error"].includes(payload.type)) {
        state.requestSentAt.delete(requestId);
    }
}

function firstFinite(values) {
    for (const value of values) {
        const number = finiteNumber(value);
        if (number !== null) return number;
    }
    return null;
}

function setConfidence(confidence) {
    const normalized = clamp(finiteNumber(confidence) ?? 0, 0, 1);
    const percentage = Math.round(normalized * 100);
    DOM.confidenceText.textContent = `${percentage}%`;
    DOM.confidenceFill.style.width = `${percentage}%`;
    DOM.confidenceProgress.setAttribute("aria-valuenow", String(percentage));
}

function clearLivePrediction(title = "No live prediction", detail = "Present a sign when the camera and model are ready.") {
    DOM.predictionText.className = "prediction-text waiting";
    DOM.predictionText.textContent = title;
    DOM.predictionDetail.textContent = detail;
    DOM.predictionState.dataset.state = "waiting";
    DOM.predictionState.textContent = "Waiting";
    setConfidence(0);
    renderTopPredictions([]);
}

function renderTopPredictions(predictions) {
    DOM.topPredictionsList.replaceChildren();
    const normalized = predictions
        .slice(0, 5)
        .map((prediction) => {
            if (!prediction || typeof prediction !== "object") return null;
            const label = safeText(prediction.label, safeText(prediction.sign, safeText(prediction.class)));
            const confidence = finiteNumber(prediction.confidence ?? prediction.probability ?? prediction.score);
            if (!label || confidence === null) return null;
            return { label, confidence: clamp(confidence, 0, 1) };
        })
        .filter(Boolean);

    if (normalized.length === 0) {
        const empty = document.createElement("li");
        empty.className = "empty-state";
        empty.textContent = "No candidates yet";
        DOM.topPredictionsList.append(empty);
        return;
    }

    const fragment = document.createDocumentFragment();
    normalized.forEach((prediction, index) => {
        const item = document.createElement("li");
        item.className = "candidate-item";

        const rank = document.createElement("span");
        rank.className = "candidate-rank";
        rank.textContent = `#${index + 1}`;

        const label = document.createElement("span");
        label.className = "candidate-label";
        label.textContent = prediction.label;

        const confidence = document.createElement("span");
        confidence.className = "candidate-confidence";
        confidence.textContent = `${Math.round(prediction.confidence * 100)}%`;

        item.append(rank, label, confidence);
        fragment.append(item);
    });
    DOM.topPredictionsList.append(fragment);
}

function addToHistory(label, confidence, mode) {
    state.history.unshift({
        label,
        confidence: clamp(finiteNumber(confidence) ?? 0, 0, 1),
        mode,
        time: new Date(),
    });
    state.history = state.history.slice(0, 50);
    renderHistory();
}

function renderHistory() {
    DOM.historyList.replaceChildren();
    if (state.history.length === 0) {
        const empty = document.createElement("li");
        empty.className = "empty-state";
        empty.textContent = "Accepted signs will appear here";
        DOM.historyList.append(empty);
        return;
    }

    const fragment = document.createDocumentFragment();
    state.history.forEach((entry) => {
        const item = document.createElement("li");
        item.className = "history-item";

        const label = document.createElement("span");
        label.className = "history-label";
        label.textContent = entry.label;

        const meta = document.createElement("span");
        meta.className = "history-meta";
        meta.textContent = `${entry.mode === "words" ? "Word" : "Letter"} | ${Math.round(entry.confidence * 100)}% | ${entry.time.toLocaleTimeString()}`;

        item.append(label, meta);
        fragment.append(item);
    });
    DOM.historyList.append(fragment);
}

function scheduleKeyboardReadinessRefresh(delayMs) {
    if (state.keyboard.cooldownTimer) window.clearTimeout(state.keyboard.cooldownTimer);
    state.keyboard.cooldownTimer = null;
    if (delayMs <= 0) return;
    state.keyboard.cooldownTimer = window.setTimeout(() => {
        state.keyboard.cooldownTimer = null;
        renderKeyboard();
    }, Math.ceil(delayMs) + 20);
}

function keyboardReadiness() {
    if (state.mode !== "alphabet") return { state: "paused", text: "Switch to Alphabet keyboard to spell" };
    if (!state.cameraRunning) return { state: "paused", text: "Start the camera to spell" };
    if (!isWebSocketOpen()) return { state: "paused", text: "Waiting for the recognition stream" };

    const elapsed = performance.now() - state.alphabet.lastAcceptedAt;
    const remainingMs = Math.max(0, LETTER_ACCEPT_COOLDOWN_MS - elapsed);
    if (remainingMs > 0) {
        return { state: "cooldown", text: `Next letter in ${(remainingMs / 1_000).toFixed(1)}s`, remainingMs };
    }
    if (!state.alphabet.armed) {
        return { state: "waiting", text: "Lower your hand or show a different letter" };
    }
    return { state: "ready", text: "Ready for a letter" };
}

function cancelSuggestionRequest() {
    if (state.keyboard.suggestionTimer) window.clearTimeout(state.keyboard.suggestionTimer);
    state.keyboard.suggestionTimer = null;
    if (state.keyboard.suggestionAbortController) state.keyboard.suggestionAbortController.abort();
    state.keyboard.suggestionAbortController = null;
}

function ensureKeyboardSuggestions() {
    const prefix = safeText(state.keyboard.spelling).toLowerCase().replace(/[^a-z]/g, "");
    const context = sentenceContextWords(state.keyboard.sentence);
    const queryKey = `${context.join(" ")}|${prefix}`;
    if (queryKey === state.keyboard.suggestionQueryKey) return;

    cancelSuggestionRequest();
    state.keyboard.suggestionQueryKey = queryKey;
    state.keyboard.selectedSuggestionIndex = 0;
    state.keyboard.suggestions = buildSuggestions(prefix, state.keyboard.vocabulary);
    state.keyboard.suggestionSource = "local";

    if (!prefix && context.length === 0) {
        state.keyboard.suggestionStatus = "idle";
        return;
    }

    state.keyboard.suggestionStatus = "loading";
    const requestNumber = ++state.keyboard.suggestionRequestNumber;
    state.keyboard.suggestionTimer = window.setTimeout(async () => {
        state.keyboard.suggestionTimer = null;
        const controller = new AbortController();
        state.keyboard.suggestionAbortController = controller;
        const timeout = window.setTimeout(() => controller.abort(), 4_000);
        try {
            const response = await fetch(CONTEXT_SUGGESTION_ENDPOINT, {
                method: "POST",
                cache: "no-store",
                headers: { "Content-Type": "application/json", Accept: "application/json" },
                body: JSON.stringify({
                    context,
                    previous_words: context,
                    sentence: state.keyboard.sentence.trim(),
                    prefix,
                    limit: MAX_SUGGESTIONS,
                }),
                signal: controller.signal,
            });
            if (!response.ok) throw new Error(`Suggestion service returned ${response.status}`);
            const payload = await response.json();
            if (requestNumber !== state.keyboard.suggestionRequestNumber
                    || queryKey !== state.keyboard.suggestionQueryKey) return;
            const contextual = normalizeContextSuggestions(payload, prefix);
            if (contextual.length) {
                state.keyboard.suggestions = contextual;
                state.keyboard.suggestionSource = "context";
            }
            state.keyboard.suggestionStatus = "ready";
        } catch (error) {
            if (error?.name !== "AbortError" && requestNumber === state.keyboard.suggestionRequestNumber) {
                state.keyboard.suggestionStatus = "fallback";
            }
        } finally {
            window.clearTimeout(timeout);
            if (state.keyboard.suggestionAbortController === controller) {
                state.keyboard.suggestionAbortController = null;
            }
            if (requestNumber === state.keyboard.suggestionRequestNumber) renderKeyboard();
        }
    }, SUGGESTION_DEBOUNCE_MS);
}

function selectedSuggestion() {
    const count = state.keyboard.suggestions.length;
    if (!count) return "";
    const index = clamp(state.keyboard.selectedSuggestionIndex, 0, count - 1);
    return state.keyboard.suggestions[index] || "";
}

function renderKeyboard() {
    ensureKeyboardSuggestions();
    if (state.keyboard.selectedSuggestionIndex >= state.keyboard.suggestions.length) {
        state.keyboard.selectedSuggestionIndex = 0;
    }
    DOM.currentLetter.textContent = state.keyboard.currentLetter || "—";
    DOM.currentSpelling.textContent = state.keyboard.spelling || "Start signing a word";
    DOM.currentSpelling.classList.toggle("is-placeholder", !state.keyboard.spelling);

    const readiness = keyboardReadiness();
    DOM.inputReadiness.dataset.state = readiness.state;
    DOM.inputReadiness.textContent = readiness.text;
    scheduleKeyboardReadinessRefresh(readiness.remainingMs || 0);

    DOM.wordSuggestions.replaceChildren();
    const hasContext = sentenceContextWords(state.keyboard.sentence).length > 0;
    if (!state.keyboard.spelling && !hasContext) {
        const empty = document.createElement("p");
        empty.className = "suggestion-empty";
        empty.textContent = "Sign a letter to begin";
        DOM.wordSuggestions.append(empty);
        DOM.suggestionHelp.textContent = state.keyboard.vocabularySource === "api"
            ? "Suggestions appear after your first letter."
            : "Using built-in suggestions while the vocabulary service reconnects.";
    } else if (state.keyboard.suggestions.length === 0) {
        const empty = document.createElement("p");
        empty.className = "suggestion-empty";
        empty.textContent = `No saved words begin with “${state.keyboard.spelling.toLowerCase()}”. Use Space to add it as typed.`;
        if (!state.keyboard.spelling) {
            empty.textContent = state.keyboard.suggestionStatus === "loading"
                ? "Predicting the next word from your sentence..."
                : "No contextual prediction is available yet. Sign the next word to continue.";
        }
        DOM.wordSuggestions.append(empty);
        DOM.suggestionHelp.textContent = state.keyboard.spelling
            ? "Keep spelling or add the letters as a custom word."
            : "Suggestions update after each completed word.";
    } else {
        const fragment = document.createDocumentFragment();
        state.keyboard.suggestions.forEach((word, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "suggestion-chip";
            button.dataset.word = word;
            button.dataset.suggestionIndex = String(index);
            button.classList.toggle("is-selected", index === state.keyboard.selectedSuggestionIndex);
            button.setAttribute("aria-selected", String(index === state.keyboard.selectedSuggestionIndex));
            button.setAttribute("aria-label", `${index === 0 ? "Best suggestion: " : "Select word "}${word}`);

            if (state.keyboard.spelling && word.startsWith(state.keyboard.spelling.toLowerCase())) {
                const match = document.createElement("strong");
                match.textContent = word.slice(0, state.keyboard.spelling.length);
                const rest = document.createTextNode(word.slice(state.keyboard.spelling.length));
                button.append(match, rest);
            } else {
                button.textContent = word;
            }
            fragment.append(button);
        });
        DOM.wordSuggestions.append(fragment);
        DOM.suggestionHelp.textContent = state.keyboard.suggestionSource === "context"
            ? `${state.keyboard.suggestions.length} context-ranked ${state.keyboard.suggestions.length === 1 ? "prediction" : "predictions"}. Use arrows, Enter, or click.`
            : `${state.keyboard.suggestions.length} matching ${state.keyboard.suggestions.length === 1 ? "word" : "words"}. Use arrows, Enter, or click.`;
    }

    const hasSpelling = Boolean(state.keyboard.spelling);
    DOM.keyboardBackspace.disabled = !hasSpelling;
    DOM.keyboardClearLetters.disabled = !hasSpelling;
    DOM.keyboardSpace.disabled = !hasSpelling;
    const suggestion = selectedSuggestion();
    DOM.keyboardConfirm.disabled = !suggestion;
    DOM.keyboardConfirm.textContent = suggestion ? `Accept “${suggestion}”` : "Accept best suggestion";
    DOM.keyboardClearSentence.disabled = !state.keyboard.sentence;
    DOM.sentenceOutput.textContent = state.keyboard.sentence || "Your accepted words will appear here.";
    DOM.sentenceOutput.classList.toggle("has-content", Boolean(state.keyboard.sentence));
}

function tryAcceptAlphabetLetter(label, confidence, now = performance.now()) {
    const transition = getLetterAcceptanceTransition({
        label,
        armed: state.alphabet.armed,
        lastAcceptedLabel: state.alphabet.lastAcceptedLabel,
        lastAcceptedAt: state.alphabet.lastAcceptedAt,
        now,
    });

    state.alphabet.armed = transition.armed;
    state.alphabet.lastAcceptedLabel = transition.lastAcceptedLabel;
    state.alphabet.lastAcceptedAt = transition.lastAcceptedAt;

    if (!transition.accepted) {
        renderKeyboard();
        return transition;
    }

    state.keyboard.currentLetter = transition.lastAcceptedLabel;
    state.keyboard.spelling += transition.lastAcceptedLabel;
    addToHistory(transition.lastAcceptedLabel, confidence, "alphabet");
    playFeedback("letter");
    renderKeyboard();
    return transition;
}

function acceptPendingDetectedLetter() {
    const letter = normalizeDetectedLetter(state.keyboard.pendingLetter);
    if (!letter || !state.keyboard.pendingLetterAcceptable || state.keyboard.pendingLetterCommitted) return false;
    const now = performance.now();
    const transition = getLetterAcceptanceTransition({
        label: letter,
        armed: state.alphabet.armed,
        lastAcceptedLabel: state.alphabet.lastAcceptedLabel,
        lastAcceptedAt: state.alphabet.lastAcceptedAt,
        now,
    });
    if (!transition.accepted) return false;
    state.alphabet.armed = transition.armed;
    state.alphabet.lastAcceptedAt = transition.lastAcceptedAt;
    state.alphabet.lastAcceptedLabel = transition.lastAcceptedLabel;
    state.keyboard.currentLetter = transition.lastAcceptedLabel;
    state.keyboard.spelling += transition.lastAcceptedLabel;
    state.keyboard.pendingLetterCommitted = true;
    addToHistory(transition.lastAcceptedLabel, state.keyboard.pendingConfidence, "alphabet");
    playFeedback("letter");
    renderKeyboard();
    return true;
}

function clearCurrentLetters() {
    state.keyboard.spelling = "";
    state.keyboard.currentLetter = "";
    state.keyboard.pendingLetterCommitted = true;
    renderKeyboard();
}

function backspaceCurrentLetters() {
    state.keyboard.spelling = state.keyboard.spelling.slice(0, -1);
    state.keyboard.currentLetter = state.keyboard.spelling.slice(-1);
    renderKeyboard();
}

function commitKeyboardWord(word = "") {
    const previousSentence = state.keyboard.sentence;
    const transition = commitWordTransition(state.keyboard, word);
    if (transition.sentence === previousSentence) return false;
    state.keyboard.sentence = transition.sentence;
    state.keyboard.spelling = transition.spelling;
    state.keyboard.currentLetter = "";
    state.keyboard.pendingLetterCommitted = true;
    playFeedback("word");
    renderKeyboard();
    return true;
}

function clearSentence() {
    state.keyboard.sentence = "";
    renderKeyboard();
}

function ensureAudioContext() {
    if (state.keyboard.audioContext) return state.keyboard.audioContext;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return null;
    state.keyboard.audioContext = new AudioContextClass();
    return state.keyboard.audioContext;
}

function primeAudioFeedback() {
    if (!state.keyboard.soundEnabled) return;
    const context = ensureAudioContext();
    if (context?.state === "suspended") context.resume().catch(() => {});
}

function playFeedback(kind) {
    if (!state.keyboard.soundEnabled) return;
    const context = ensureAudioContext();
    if (!context) return;
    if (context.state === "suspended") {
        context.resume().then(() => playFeedback(kind)).catch(() => {});
        return;
    }
    if (context.state !== "running") return;

    const start = context.currentTime;
    const notes = kind === "word"
        ? [{ frequency: 520, offset: 0, duration: 0.07 }, { frequency: 780, offset: 0.085, duration: 0.11 }]
        : [{ frequency: 660, offset: 0, duration: 0.075 }];

    notes.forEach((note) => {
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.type = kind === "word" ? "sine" : "triangle";
        oscillator.frequency.setValueAtTime(note.frequency, start + note.offset);
        gain.gain.setValueAtTime(0.0001, start + note.offset);
        gain.gain.exponentialRampToValueAtTime(0.09, start + note.offset + 0.012);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + note.offset + note.duration);
        oscillator.connect(gain);
        gain.connect(context.destination);
        oscillator.start(start + note.offset);
        oscillator.stop(start + note.offset + note.duration + 0.015);
    });
}

function toggleSound() {
    state.keyboard.soundEnabled = !state.keyboard.soundEnabled;
    DOM.soundToggle.setAttribute("aria-pressed", String(state.keyboard.soundEnabled));
    DOM.soundToggle.setAttribute("aria-label", `Turn sound feedback ${state.keyboard.soundEnabled ? "off" : "on"}`);
    DOM.soundToggleLabel.textContent = state.keyboard.soundEnabled ? "Sound on" : "Sound off";
    DOM.soundToggle.classList.toggle("is-muted", !state.keyboard.soundEnabled);
    if (state.keyboard.soundEnabled) {
        primeAudioFeedback();
        playFeedback("letter");
    }
}

async function fetchVocabulary() {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5_000);
    try {
        const response = await fetch("/api/vocabulary", {
            cache: "no-store",
            headers: { Accept: "application/json" },
            signal: controller.signal,
        });
        if (!response.ok) throw new Error("Vocabulary unavailable");
        const payload = await response.json();
        const vocabulary = Array.isArray(payload?.vocabulary)
            ? payload.vocabulary.map((word) => safeText(word).toLowerCase()).filter(Boolean)
            : [];
        if (vocabulary.length === 0) throw new Error("Vocabulary is empty");
        state.keyboard.vocabulary = vocabulary;
        state.keyboard.vocabularySource = "api";
    } catch (error) {
        state.keyboard.vocabulary = [...FALLBACK_VOCABULARY];
        state.keyboard.vocabularySource = "fallback";
    } finally {
        window.clearTimeout(timeout);
        state.keyboard.suggestionQueryKey = "";
        renderKeyboard();
    }
}

function pageFromHash() {
    return window.location.hash === "#text-to-sign" ? "text-to-sign" : "sign-input";
}

function setActivePage(page, updateHash = true) {
    const nextPage = page === "text-to-sign" ? "text-to-sign" : "sign-input";
    if (nextPage !== "sign-input" && state.cameraRunning) stopCamera(true);
    if (nextPage !== "text-to-sign" && isVoiceInputActive()) {
        stopRecording(false);
    }
    state.activePage = nextPage;
    DOM.pageSignInput.hidden = nextPage !== "sign-input";
    DOM.pageTextToSign.hidden = nextPage !== "text-to-sign";
    DOM.navSignInput.setAttribute("aria-pressed", String(nextPage === "sign-input"));
    DOM.navTextToSign.setAttribute("aria-pressed", String(nextPage === "text-to-sign"));
    document.title = nextPage === "sign-input"
        ? "SignBridge | Sign to text"
        : "SignBridge | Text & voice to sign";
    if (updateHash) {
        const hash = nextPage === "text-to-sign" ? "#text-to-sign" : "#sign-input";
        if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
    }
}

function updateTranslationCharacterCount() {
    const count = DOM.translationInput.value.length;
    DOM.translationCharCount.textContent = `${count} / 2000`;
    updateTranslationControls();
}

function updateTranslationControls() {
    const recording = isVoiceInputActive();
    const hasText = Boolean(DOM.translationInput.value.trim());
    const browserCanRecognize = Boolean(speechRecognitionClass());
    const browserCanRecord = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia);
    const canUseRecordedAudio = browserCanRecord && state.capabilities.speechToSign.voiceReady;
    const canUseBrowserSpeech = browserCanRecognize && state.capabilities.speechToSign.textReady
        && (!state.translation.preferRecordedAudio || !canUseRecordedAudio);
    const voiceMode = canUseRecordedAudio && state.translation.preferRecordedAudio
        ? "recorded"
        : canUseBrowserSpeech ? "browser" : canUseRecordedAudio ? "recorded" : "none";
    DOM.translateText.disabled = state.translation.busy || recording || !hasText
        || !state.capabilities.speechToSign.textReady;
    DOM.clearTranslationInput.disabled = state.translation.busy || recording || DOM.translationInput.value.length === 0;
    DOM.startRecording.disabled = state.translation.busy || recording || voiceMode === "none";
    DOM.stopRecording.disabled = !recording;
    DOM.translateText.title = state.capabilities.speechToSign.textReady
        ? ""
        : "Text translation service is unavailable";
    DOM.startRecording.title = voiceMode !== "none"
        ? ""
        : !browserCanRecognize && !browserCanRecord
            ? "This browser does not expose a supported microphone API"
            : "Voice transcription is unavailable";
    DOM.voiceInputMode.textContent = voiceMode === "browser"
        ? "Browser recognition — provider may process audio"
        : voiceMode === "recorded"
            ? "Local Whisper transcription"
            : "Voice input unavailable";
    DOM.startRecordingLabel.textContent = voiceMode === "recorded" ? "Start recording" : "Start listening";
    DOM.stopRecordingLabel.textContent = voiceMode === "recorded" ? "Stop & translate" : "Stop listening";
    DOM.voicePrivacyNote.textContent = voiceMode === "browser"
        ? "Your browser may send audio to its speech provider. Clicking Start listening authorizes that browser service."
        : "";
    DOM.voicePrivacyNote.classList.toggle("is-hidden", voiceMode !== "browser");
}

function setTranslationState(label, status) {
    DOM.translationState.textContent = label;
    DOM.translationState.dataset.state = status;
}

function setTranslationMessage(message = "", status = "info") {
    DOM.translationMessage.textContent = message;
    DOM.translationMessage.dataset.state = status;
    DOM.translationMessage.classList.toggle("is-hidden", !message);
}

function setTranslationBusy(busy, message = "") {
    state.translation.busy = busy;
    setTranslationState(busy ? "Working" : "Ready", busy ? "loading" : "waiting");
    if (message) setTranslationMessage(message);
    updateTranslationControls();
}

async function apiErrorMessage(response) {
    try {
        const payload = await response.json();
        const detail = payload?.detail;
        const code = safeText(detail?.code, safeText(payload?.code));
        const message = safeText(detail?.message, typeof detail === "string" ? detail : safeText(payload?.message));
        if (code === "speech_model_missing") {
            return "Voice transcription is not installed on this server yet. You can still type and translate a sentence.";
        }
        return message || `The server returned ${response.status}.`;
    } catch (error) {
        return `The server returned ${response.status}.`;
    }
}

async function requestTranslation(endpoint, options, loadingMessage) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), TRANSLATION_REQUEST_TIMEOUT_MS);
    setTranslationBusy(true, loadingMessage);
    try {
        const response = await fetch(endpoint, { ...options, signal: controller.signal });
        if (!response.ok) throw new Error(await apiErrorMessage(response));
        const payload = await response.json();
        if (!payload || typeof payload !== "object") throw new Error("The server returned an invalid translation.");
        renderTranslationResult(payload);
        return payload;
    } catch (error) {
        const message = error?.name === "AbortError"
            ? "Translation timed out. Check the server and try again."
            : safeText(error?.message, "Translation failed. Please try again.");
        setTranslationState("Error", "error");
        setTranslationMessage(message, "error");
        return null;
    } finally {
        window.clearTimeout(timeout);
        state.translation.busy = false;
        updateTranslationControls();
    }
}

async function translateTypedText() {
    const text = DOM.translationInput.value.trim();
    if (!text || state.translation.busy || !state.capabilities.speechToSign.textReady) return null;
    return requestTranslation(
        "/api/text-to-sign",
        {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({ text }),
        },
        "Understanding the sentence and preparing its sign sequence…",
    );
}

function normalizedClipWord(clip) {
    return safeText(clip?.gloss, safeText(clip?.word, "Sign"));
}

function safeMediaUrl(value) {
    const raw = safeText(value);
    if (!raw) return "";
    try {
        const url = new URL(raw, window.location.origin);
        if (!/^https?:$/.test(url.protocol)) return "";
        return url.href;
    } catch (error) {
        return "";
    }
}

function clipMediaUrl(clip) {
    return safeMediaUrl(
        clip?.path ?? clip?.url ?? clip?.src ?? clip?.media_url
        ?? clip?.asset_url ?? clip?.video_url ?? clip?.image_url,
    );
}

function clipMediaKind(clip, mediaUrl = clipMediaUrl(clip)) {
    if (!mediaUrl) return "";
    const declared = safeText(clip?.mime_type, safeText(clip?.media_type)).toLowerCase();
    const type = safeText(clip?.type, safeText(clip?.representation)).toLowerCase();
    let pathname = "";
    try {
        pathname = new URL(mediaUrl).pathname.toLowerCase();
    } catch (error) {
        pathname = mediaUrl.toLowerCase();
    }
    if (declared.startsWith("image/") || ["image", "gif", "photo"].includes(type)
            || /\.(gif|png|jpe?g|webp|avif|svg)$/.test(pathname)) return "image";
    if (declared.startsWith("video/") || ["video", "clip", "native_sign_video"].includes(type)
            || /\.(mp4|webm|mov|m4v|ogv)$/.test(pathname)) return "video";
    return type.includes("image") ? "image" : "video";
}

function translationClipsFromPayload(payload, tokens = []) {
    const candidates = [payload?.sign_clips, payload?.clips, payload?.signs, payload?.sequence, payload?.visuals];
    let raw = candidates.find((candidate) => Array.isArray(candidate)) || [];
    if (!raw.length) raw = tokens;
    return raw.map((item, index) => {
        if (typeof item === "string") {
            return { gloss: item, word: item, type: "fingerspell", letters: [...item], sequence_index: index };
        }
        if (!item || typeof item !== "object") return null;
        const nested = item.sign && typeof item.sign === "object" ? item.sign : {};
        return { ...nested, ...item, sequence_index: finiteNumber(item.sequence_index) ?? index };
    }).filter(Boolean);
}

function fingerspellLetters(clip) {
    const provided = Array.isArray(clip?.letters) ? clip.letters : [];
    const letters = provided.map((letter) => normalizeDetectedLetter(letter)).filter(Boolean);
    if (letters.length) return letters;
    return [...normalizedClipWord(clip).toUpperCase()].filter((letter) => /^[A-Z]$/.test(letter));
}

function renderFingerspellVisual(container, clip, fallback = false) {
    container.replaceChildren();
    container.className = "sign-visual fingerspell-visual";
    const letters = fingerspellLetters(clip);
    if (letters.length === 0) {
        const unavailable = document.createElement("p");
        unavailable.className = "sign-unavailable";
        unavailable.textContent = "No visual representation is available for this concept.";
        container.append(unavailable);
        return;
    }
    const steps = Array.isArray(clip?.letter_steps) ? clip.letter_steps : [];
    const hasGuides = steps.some((step) => safeMediaUrl(step?.path));
    const label = document.createElement("span");
    label.className = "fingerspell-label";
    label.textContent = hasGuides
        ? fallback ? "Native video unavailable — fingerspelling landmark guides" : "Fingerspelling landmark guides"
        : fallback ? "Native video unavailable — letter prompts" : "Fingerspelling letter prompts";
    const row = document.createElement("div");
    row.className = hasGuides ? "fingerspell-guides" : "letter-tiles";
    row.setAttribute("aria-label", `Fingerspell ${letters.join(", ")}`);
    letters.forEach((letter, index) => {
        const step = steps[index] && typeof steps[index] === "object" ? steps[index] : {};
        const guideUrl = safeMediaUrl(step.path);
        if (guideUrl) {
            const guide = document.createElement("figure");
            guide.className = "fingerspell-guide";
            const image = document.createElement("img");
            image.loading = index < 4 ? "eager" : "lazy";
            image.decoding = "async";
            image.src = guideUrl;
            image.alt = `${letter} fingerspelling landmark guide`;
            const caption = document.createElement("figcaption");
            caption.textContent = letter;
            image.addEventListener("error", () => {
                guide.classList.add("is-unavailable");
                image.remove();
                caption.textContent = `${letter} guide unavailable`;
            }, { once: true });
            guide.append(image, caption);
            row.append(guide);
            return;
        }
        const tile = document.createElement("span");
        tile.className = `letter-tile${step.motion_required ? " requires-motion" : ""}`;
        tile.textContent = letter;
        if (step.motion_required) {
            tile.title = `${letter} requires motion and cannot be represented by a static guide`;
            tile.setAttribute("aria-label", `${letter}, motion required; static guide unavailable`);
        }
        row.append(tile);
    });
    container.append(label, row);
    if (steps.some((step) => step?.motion_required)) {
        const warning = document.createElement("p");
        warning.className = "motion-guide-warning";
        warning.textContent = "J and Z require motion; a static guide would be misleading.";
        container.append(warning);
    }
}

function createSignSequenceItem(clip, index) {
    const item = document.createElement("li");
    item.className = "sign-sequence-item";
    item.dataset.sequenceIndex = String(index);

    const heading = document.createElement("div");
    heading.className = "sign-card-heading";
    const number = document.createElement("span");
    number.className = "sequence-number";
    number.textContent = String(index + 1).padStart(2, "0");
    const word = document.createElement("h3");
    word.textContent = normalizedClipWord(clip);
    const kind = document.createElement("span");
    kind.className = "sign-kind";
    const mediaUrl = clipMediaUrl(clip);
    const mediaKind = clipMediaKind(clip, mediaUrl);
    kind.textContent = mediaUrl ? (mediaKind === "image" ? "Sign visual" : "Native sign") : "Fingerspelled";
    heading.append(number, word, kind);

    const visual = document.createElement("div");
    if (mediaUrl && mediaKind === "video") {
        visual.className = "sign-visual video-visual";
        const video = document.createElement("video");
        video.controls = true;
        video.playsInline = true;
        video.muted = true;
        video.preload = "metadata";
        video.src = mediaUrl;
        video.setAttribute("aria-label", `${normalizedClipWord(clip)} sign video`);
        video.addEventListener("error", () => {
            kind.textContent = "Fingerspelled fallback";
            renderFingerspellVisual(visual, clip, true);
        }, { once: true });
        visual.append(video);
    } else if (mediaUrl && mediaKind === "image") {
        visual.className = "sign-visual image-visual";
        const image = document.createElement("img");
        image.loading = index < 3 ? "eager" : "lazy";
        image.decoding = "async";
        image.src = mediaUrl;
        image.alt = `${normalizedClipWord(clip)} sign visual`;
        image.addEventListener("error", () => {
            kind.textContent = "Fingerspelled fallback";
            renderFingerspellVisual(visual, clip, true);
        }, { once: true });
        visual.append(image);
    } else {
        renderFingerspellVisual(visual, clip);
    }

    const fallbackReason = safeText(clip?.fallback_reason);
    if (fallbackReason && !mediaUrl) {
        const note = document.createElement("p");
        note.className = "sign-card-note";
        note.textContent = fallbackReason.replaceAll("_", " ");
        item.append(heading, visual, note);
    } else {
        item.append(heading, visual);
    }
    return item;
}

function renderTranslationResult(payload) {
    const transcript = safeText(payload.transcript, safeText(payload.original_text));
    const tokens = Array.isArray(payload.normalized_tokens)
        ? payload.normalized_tokens.map((token) => safeText(token)).filter(Boolean)
        : [];
    const clips = translationClipsFromPayload(payload, tokens);

    if (transcript) {
        DOM.translationInput.value = transcript.slice(0, 2_000);
        updateTranslationCharacterCount();
    }

    DOM.translationEmpty.classList.add("is-hidden");
    DOM.translationSummary.classList.remove("is-hidden");
    DOM.translationTranscript.textContent = transcript || "No transcript returned";
    DOM.normalizedTokens.replaceChildren();
    if (tokens.length) {
        tokens.forEach((token) => {
            const chip = document.createElement("span");
            chip.className = "token-chip";
            chip.textContent = token;
            DOM.normalizedTokens.append(chip);
        });
    } else {
        const empty = document.createElement("span");
        empty.className = "token-empty";
        empty.textContent = "No sign concepts were produced.";
        DOM.normalizedTokens.append(empty);
    }

    const statusNote = safeText(payload.translation_status?.message);
    const coverage = payload.coverage && typeof payload.coverage === "object" ? payload.coverage : null;
    const nativeCount = finiteNumber(coverage?.native_signs);
    const totalConcepts = finiteNumber(coverage?.total_concepts);
    const fingerspelled = finiteNumber(coverage?.fingerspelled);
    const nativeRatio = finiteNumber(coverage?.native_ratio);
    const coverageNote = nativeCount === null
        ? ""
        : [
            `${Math.round(nativeCount)} native ${Math.round(nativeCount) === 1 ? "sign" : "signs"}`,
            fingerspelled === null ? "" : `${Math.round(fingerspelled)} fingerspelled`,
            nativeRatio === null ? "" : `${Math.round(clamp(nativeRatio, 0, 1) * 100)}% native coverage`,
            totalConcepts === null ? "" : `${Math.round(totalConcepts)} total ${Math.round(totalConcepts) === 1 ? "concept" : "concepts"}`,
        ].filter(Boolean).join(" · ");
    DOM.translationNote.textContent = [statusNote, coverageNote].filter(Boolean).join(" ");
    DOM.translationNote.classList.toggle("is-hidden", !DOM.translationNote.textContent);

    DOM.signSequence.replaceChildren();
    clips.forEach((clip, index) => DOM.signSequence.append(createSignSequenceItem(clip, index)));
    state.translation.clips = clips;
    state.translation.activeIndex = 0;
    DOM.signCount.textContent = `${clips.length} ${clips.length === 1 ? "step" : "steps"}`;
    DOM.signResultsCard.classList.toggle("is-hidden", clips.length === 0);
    setActiveSequenceIndex(0);

    if (clips.length === 0) {
        setTranslationState("Empty", "uncertain");
        setTranslationMessage("No renderable sign concepts were found. Try a sentence with more descriptive words.", "warning");
    } else {
        setTranslationState("Complete", "accepted");
        setTranslationMessage("");
    }
}

function setActiveSequenceIndex(index, scroll = false) {
    const count = state.translation.clips.length;
    if (count === 0) {
        state.translation.activeIndex = 0;
        DOM.sequencePosition.textContent = "0 / 0";
        DOM.previousSign.disabled = true;
        DOM.nextSign.disabled = true;
        DOM.sequenceAction.disabled = true;
        DOM.sequenceAction.textContent = "Play sign";
        return;
    }

    const nextIndex = clamp(Math.round(finiteNumber(index) ?? 0), 0, count - 1);
    state.translation.activeIndex = nextIndex;
    const items = [...DOM.signSequence.querySelectorAll(".sign-sequence-item")];
    items.forEach((item, itemIndex) => {
        const active = itemIndex === nextIndex;
        item.classList.toggle("is-active", active);
        if (active) item.setAttribute("aria-current", "step");
        else item.removeAttribute("aria-current");
    });
    DOM.sequencePosition.textContent = `${nextIndex + 1} / ${count}`;
    DOM.previousSign.disabled = nextIndex === 0;
    DOM.nextSign.disabled = nextIndex === count - 1;
    const activeVideo = items[nextIndex]?.querySelector("video");
    DOM.sequenceAction.disabled = false;
    DOM.sequenceAction.textContent = activeVideo
        ? "Play sign"
        : nextIndex === count - 1 ? "Replay sequence" : "Next sign";
    if (scroll) items[nextIndex]?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
}

function moveSequence(direction) {
    setActiveSequenceIndex(state.translation.activeIndex + direction, true);
}

function activateSequenceAction() {
    const count = state.translation.clips.length;
    if (!count) return;
    const items = [...DOM.signSequence.querySelectorAll(".sign-sequence-item")];
    const activeItem = items[state.translation.activeIndex];
    const video = activeItem?.querySelector("video");
    if (!video) {
        const nextIndex = state.translation.activeIndex === count - 1 ? 0 : state.translation.activeIndex + 1;
        setActiveSequenceIndex(nextIndex, true);
        return;
    }

    items.forEach((item) => {
        const otherVideo = item.querySelector("video");
        if (otherVideo && otherVideo !== video) otherVideo.pause();
    });
    video.currentTime = 0;
    video.play().catch(() => {
        setTranslationMessage("The sign video could not autoplay. Use the video’s own Play control.", "warning");
    });
    if (state.translation.activeIndex < count - 1) {
        video.addEventListener("ended", () => moveSequence(1), { once: true });
    }
}

function speechRecognitionClass() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function isVoiceInputActive() {
    return state.translation.recognitionActive || state.translation.mediaRecorder?.state === "recording";
}

function combinedRecognitionText(finalText, interimText = "") {
    return [safeText(state.translation.recognitionBaseText), safeText(finalText), safeText(interimText)]
        .filter(Boolean)
        .join(" ")
        .replace(/\s+/g, " ")
        .slice(0, 2_000);
}

function renderLiveRecognition(finalText = "", interimText = "") {
    const combined = combinedRecognitionText(finalText, interimText);
    DOM.translationInput.value = combined;
    updateTranslationCharacterCount();
    DOM.voiceLiveTranscript.replaceChildren();
    if (!finalText && !interimText) {
        DOM.voiceLiveTranscript.classList.add("is-hidden");
        return;
    }
    DOM.voiceLiveTranscript.classList.remove("is-hidden");
    const finalSpan = document.createElement("span");
    finalSpan.textContent = safeText(finalText);
    if (interimText) {
        const interimSpan = document.createElement("span");
        interimSpan.className = "voice-interim";
        interimSpan.textContent = `${finalText ? " " : ""}${safeText(interimText)}`;
        finalSpan.append(interimSpan);
    }
    DOM.voiceLiveTranscript.append(finalSpan);
}

function finishBrowserRecognition(shouldTranslate = true) {
    const finalText = safeText(state.translation.recognitionFinalTranscript);
    const interimText = safeText(state.translation.recognitionInterimTranscript);
    const recognized = finalText || interimText;
    const error = state.translation.recognitionError;
    state.translation.recognition = null;
    state.translation.recognitionActive = false;
    state.translation.recognitionStopRequested = false;
    clearRecordingTimers();
    DOM.recordingTime.textContent = "00:00";
    DOM.recordingTime.setAttribute("aria-hidden", "true");
    updateTranslationControls();

    if (!shouldTranslate || error) return;
    if (!recognized) {
        DOM.voiceStatus.textContent = "No speech was recognized. Try again and speak clearly after listening starts.";
        return;
    }
    renderLiveRecognition(recognized, "");
    DOM.voiceStatus.textContent = "Speech recognized. Translating automatically...";
    translateTypedText().then((payload) => {
        DOM.voiceStatus.textContent = payload
            ? "Speech translated. You can edit the text or listen again."
            : "Speech was recognized, but sign translation failed. Try Translate to signs again.";
    });
}

function recognitionErrorMessage(code) {
    const messages = {
        "not-allowed": "Microphone permission was denied. Allow microphone access in browser settings, then try again.",
        "service-not-allowed": "Browser speech recognition is blocked. Use browser settings or the recorded-audio fallback.",
        "audio-capture": "No working microphone was found.",
        "no-speech": "No speech was heard. Try again and speak after listening starts.",
        network: "The browser speech service is unavailable. Recorded-audio mode will be used on the next attempt.",
        "language-not-supported": "Speech recognition does not support the selected browser language.",
        aborted: "Listening stopped.",
    };
    return messages[code] || "Speech recognition failed. Try again or use typed text.";
}

function startBrowserRecognition() {
    const Recognition = speechRecognitionClass();
    if (!Recognition || state.translation.recognitionActive) return;
    const recognition = new Recognition();
    recognition.lang = document.documentElement.lang || navigator.language || "en-US";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    state.translation.recognition = recognition;
    state.translation.recognitionActive = true;
    state.translation.recognitionStopRequested = false;
    state.translation.recognitionFinalTranscript = "";
    state.translation.recognitionInterimTranscript = "";
    state.translation.recognitionError = "";
    state.translation.recognitionBaseText = DOM.translationInput.value.trim();
    DOM.voiceLiveTranscript.classList.add("is-hidden");
    DOM.voiceStatus.textContent = "Requesting microphone permission...";

    recognition.onstart = () => {
        state.translation.recordingStartedAt = Date.now();
        state.translation.recordingTimer = window.setInterval(updateRecordingClock, 250);
        state.translation.maxRecordingTimer = window.setTimeout(() => stopRecording(true), MAX_RECORDING_MS);
        DOM.recordingTime.setAttribute("aria-hidden", "false");
        DOM.voiceStatus.textContent = "Listening. Speak naturally; the final sentence will translate automatically.";
        updateRecordingClock();
        updateTranslationControls();
    };
    recognition.onresult = (event) => {
        let finalText = "";
        let interimText = "";
        for (let index = 0; index < event.results.length; index += 1) {
            const transcript = safeText(event.results[index]?.[0]?.transcript);
            if (event.results[index].isFinal) finalText += `${transcript} `;
            else interimText += `${transcript} `;
        }
        state.translation.recognitionFinalTranscript = finalText.trim();
        state.translation.recognitionInterimTranscript = interimText.trim();
        renderLiveRecognition(finalText, interimText);
        DOM.voiceStatus.textContent = interimText ? "Listening... interim words are shown in lighter text." : "Speech recognized. Finishing...";
    };
    recognition.onerror = (event) => {
        const code = safeText(event?.error, "unknown");
        if (code === "aborted" && state.translation.recognitionStopRequested) return;
        state.translation.recognitionError = code;
        DOM.voiceStatus.textContent = recognitionErrorMessage(code);
        if (code === "network" || code === "service-not-allowed") {
            state.translation.preferRecordedAudio = state.capabilities.speechToSign.voiceReady;
        }
    };
    recognition.onend = () => finishBrowserRecognition(true);
    try {
        recognition.start();
        updateTranslationControls();
    } catch (error) {
        state.translation.recognitionError = "start-failed";
        state.translation.recognitionActive = false;
        state.translation.recognition = null;
        DOM.voiceStatus.textContent = "Speech recognition could not start. Wait a moment and try again.";
        updateTranslationControls();
    }
}

function supportedRecordingMimeType() {
    if (!window.MediaRecorder) return "";
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
    ];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function recordingExtension(mimeType) {
    if (mimeType.includes("ogg")) return "ogg";
    if (mimeType.includes("mp4")) return "m4a";
    return "webm";
}

function stopMicrophoneTracks() {
    if (state.translation.micStream) {
        state.translation.micStream.getTracks().forEach((track) => track.stop());
        state.translation.micStream = null;
    }
}

function clearRecordingTimers() {
    if (state.translation.recordingTimer) window.clearInterval(state.translation.recordingTimer);
    if (state.translation.maxRecordingTimer) window.clearTimeout(state.translation.maxRecordingTimer);
    state.translation.recordingTimer = null;
    state.translation.maxRecordingTimer = null;
}

function updateRecordingClock() {
    const elapsedSeconds = Math.max(0, Math.floor((Date.now() - state.translation.recordingStartedAt) / 1_000));
    const minutes = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
    const seconds = String(elapsedSeconds % 60).padStart(2, "0");
    DOM.recordingTime.textContent = `${minutes}:${seconds}`;
}

async function startRecordedAudio() {
    if (state.translation.busy || state.translation.mediaRecorder?.state === "recording") return;
    if (!state.capabilities.speechToSign.voiceReady) {
        DOM.voiceStatus.textContent = "Voice transcription is unavailable on this server. Type your message instead.";
        return;
    }
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        DOM.voiceStatus.textContent = "Microphone recording requires HTTPS or localhost in a supported browser.";
        return;
    }

    DOM.voiceStatus.textContent = "Requesting microphone permission…";
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
            video: false,
        });
        const mimeType = supportedRecordingMimeType();
        const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        state.translation.micStream = stream;
        state.translation.mediaRecorder = recorder;
        state.translation.chunks = [];
        state.translation.shouldTranslateRecording = false;

        recorder.addEventListener("dataavailable", (event) => {
            if (event.data?.size) state.translation.chunks.push(event.data);
        });
        recorder.addEventListener("error", () => {
            DOM.voiceStatus.textContent = "Recording failed. Check microphone access and try again.";
            stopMicrophoneTracks();
            clearRecordingTimers();
            state.translation.mediaRecorder = null;
            updateTranslationControls();
        }, { once: true });
        recorder.addEventListener("stop", async () => {
            const shouldTranslate = state.translation.shouldTranslateRecording;
            const recordedType = recorder.mimeType || mimeType || "audio/webm";
            const blob = new Blob(state.translation.chunks, { type: recordedType });
            state.translation.mediaRecorder = null;
            state.translation.chunks = [];
            stopMicrophoneTracks();
            clearRecordingTimers();
            DOM.recordingTime.textContent = "00:00";
            updateTranslationControls();
            if (!shouldTranslate) {
                DOM.voiceStatus.textContent = "Recording cancelled when you left the page.";
                return;
            }
            if (!blob.size) {
                DOM.voiceStatus.textContent = "No audio was captured. Try recording again.";
                return;
            }
            DOM.voiceStatus.textContent = "Uploading audio for transcription and translation…";
            await translateRecordedAudio(blob, recordedType);
        }, { once: true });

        recorder.start(250);
        state.translation.recordingStartedAt = Date.now();
        state.translation.recordingTimer = window.setInterval(updateRecordingClock, 250);
        state.translation.maxRecordingTimer = window.setTimeout(() => {
            DOM.voiceStatus.textContent = "Maximum recording length reached. Translating now…";
            stopRecording(true);
        }, MAX_RECORDING_MS);
        DOM.voiceStatus.textContent = "Recording. Speak naturally, then choose Stop & translate.";
        DOM.recordingTime.setAttribute("aria-hidden", "false");
        updateRecordingClock();
        updateTranslationControls();
    } catch (error) {
        stopMicrophoneTracks();
        const name = safeText(error?.name);
        DOM.voiceStatus.textContent = name === "NotAllowedError" || name === "SecurityError"
            ? "Microphone permission was denied. Enable it in browser settings to use voice input."
            : name === "NotFoundError"
                ? "No microphone was found."
                : "The microphone could not start.";
        updateTranslationControls();
    }
}

function stopRecordedAudio(shouldTranslate = true) {
    const recorder = state.translation.mediaRecorder;
    if (!recorder || recorder.state !== "recording") return;
    state.translation.shouldTranslateRecording = shouldTranslate;
    DOM.voiceStatus.textContent = shouldTranslate ? "Finishing the recording…" : "Stopping microphone…";
    recorder.stop();
    updateTranslationControls();
}

function startRecording() {
    if (state.translation.busy || isVoiceInputActive()) return;
    const canRecordLocally = Boolean(window.MediaRecorder && navigator.mediaDevices?.getUserMedia
        && state.capabilities.speechToSign.voiceReady);
    if (canRecordLocally && state.translation.preferRecordedAudio) {
        startRecordedAudio();
        return;
    }
    if (speechRecognitionClass() && state.capabilities.speechToSign.textReady) {
        startBrowserRecognition();
        return;
    }
    if (canRecordLocally) startRecordedAudio();
}

function stopRecording(shouldTranslate = true) {
    if (state.translation.recognitionActive && state.translation.recognition) {
        state.translation.recognitionStopRequested = true;
        DOM.voiceStatus.textContent = shouldTranslate ? "Finishing speech recognition..." : "Stopping microphone...";
        if (!shouldTranslate) state.translation.recognitionError = "cancelled";
        try {
            if (shouldTranslate) state.translation.recognition.stop();
            else state.translation.recognition.abort();
        } catch (error) {
            finishBrowserRecognition(shouldTranslate);
        }
        return;
    }
    stopRecordedAudio(shouldTranslate);
}

async function translateRecordedAudio(blob, mimeType) {
    const form = new FormData();
    form.append("audio", blob, `signbridge-recording.${recordingExtension(mimeType)}`);
    const payload = await requestTranslation(
        "/api/speech-to-sign",
        { method: "POST", headers: { Accept: "application/json" }, body: form },
        "Transcribing your voice and preparing the sign sequence…",
    );
    DOM.voiceStatus.textContent = payload
        ? "Voice translated. You can edit the transcript and translate again."
        : "Voice translation could not be completed.";
}

function clearTranslationInput() {
    DOM.translationInput.value = "";
    updateTranslationCharacterCount();
    DOM.translationInput.focus();
}

function renderMode() {
    const alphabetSelected = state.mode === "alphabet";
    const wordsSelected = state.mode === "words";
    DOM.modeAlphabet.setAttribute("aria-pressed", String(alphabetSelected));
    DOM.modeWords.setAttribute("aria-pressed", String(wordsSelected));
    DOM.wordsCapturePanel.classList.toggle("is-hidden", !wordsSelected);

    if (alphabetSelected) {
        DOM.modeGuidance.textContent = "Accepted letters enter the keyboard after a one-second pause. Lower your hand before repeating the same letter.";
    } else if (wordsSelected) {
        DOM.modeGuidance.textContent = "Experimental isolated-word mode. Capture exactly one complete sign, then reset.";
    } else {
        DOM.modeGuidance.textContent = "Waiting for an available recognition model.";
    }
}

function setMode(mode, announce = true) {
    if (mode !== "alphabet" && mode !== "words") return;
    if (!state.capabilities[mode]?.ready) return;

    if (state.mode && state.mode !== mode) resetCurrentSign(true);
    state.mode = mode;
    state.alphabet.armed = true;
    if (mode === "alphabet") clearOverlay();
    renderMode();
    clearLivePrediction(
        mode === "alphabet" ? "Alphabet mode ready" : "Word capture ready",
        mode === "alphabet"
            ? "Start the camera and present one letter."
            : "Start the camera, then use Capture 50 frames for one isolated sign.",
    );
    updateControls();
    if (mode === "words" && !state.holistic) void prepareWordTracking();
    if (announce) setSystemMessage(mode === "words" ? "Experimental isolated-word mode selected." : "Alphabet mode selected.");
    renderKeyboard();
}

function updateControls() {
    DOM.startCamera.disabled = state.cameraStarting || state.cameraRunning;
    DOM.stopCamera.disabled = !state.cameraRunning;
    DOM.startWordCapture.disabled = !(
        state.mode === "words"
        && state.capabilities.words.ready
        && state.cameraRunning
        && state.wordTracker.status === "ready"
        && Boolean(state.holistic)
        && isWebSocketOpen()
        && !state.words.active
        && !state.words.waiting
        && !state.words.needsReset
    );
}

async function initializeHolistic() {
    if (state.holistic) return state.holistic;
    if (wordTrackerInitializationPromise) return wordTrackerInitializationPromise;

    wordTrackerInitializationPromise = (async () => {
        await loadWordTrackingLibraries();
        if (typeof window.Holistic !== "function") {
            throw new Error("The local hand-tracking library did not load. Reload the page and try again.");
        }

        const holistic = new window.Holistic({
            locateFile: (file) => `${MEDIAPIPE_HOLISTIC_BASE}/${file}`,
        });
        holistic.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            refineFaceLandmarks: false,
            minDetectionConfidence: 0.55,
            minTrackingConfidence: 0.55,
        });
        holistic.onResults(onHolisticResults);

        try {
            if (typeof holistic.initialize === "function") await holistic.initialize();
            state.holistic = holistic;
            return holistic;
        } catch (error) {
            if (typeof holistic.close === "function") holistic.close();
            throw error;
        }
    })().finally(() => {
        wordTrackerInitializationPromise = null;
    });
    return wordTrackerInitializationPromise;
}

function loadScriptOnce(url) {
    const existing = document.querySelector(`script[data-signbridge-src="${url}"]`);
    if (existing?.dataset.loaded === "true") return Promise.resolve();
    if (existing) {
        return new Promise((resolve, reject) => {
            existing.addEventListener("load", resolve, { once: true });
            existing.addEventListener("error", () => reject(new Error(`Could not load ${url}`)), { once: true });
        });
    }
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.async = true;
        script.dataset.signbridgeSrc = url;
        script.addEventListener("load", () => {
            script.dataset.loaded = "true";
            resolve();
        }, { once: true });
        script.addEventListener("error", () => reject(new Error("A local word-tracker asset is unavailable.")), { once: true });
        document.head.append(script);
    });
}

function loadWordTrackingLibraries() {
    if (!wordTrackingLibrariesPromise) {
        wordTrackingLibrariesPromise = Promise.all([
            loadScriptOnce(MEDIAPIPE_DRAWING_URL),
            loadScriptOnce(MEDIAPIPE_HOLISTIC_URL),
        ]).catch((error) => {
            wordTrackingLibrariesPromise = null;
            throw error;
        });
    }
    return wordTrackingLibrariesPromise;
}

async function prepareWordTracking() {
    if (state.holistic) {
        state.wordTracker.status = "ready";
        state.wordTracker.error = "";
        renderCapabilities();
        updateControls();
        return true;
    }
    state.wordTracker.status = "loading";
    state.wordTracker.error = "";
    DOM.captureGuidance.textContent = "Loading the experimental word landmark tracker.";
    renderCapabilities();
    try {
        await initializeHolistic();
        state.wordTracker.status = "ready";
        state.wordTracker.error = "";
        DOM.captureGuidance.textContent = "Hold your upper body and hands in view, then capture the sign once.";
        renderCapabilities();
        return true;
    } catch (error) {
        const message = safeText(error?.message, "The word landmark tracker could not load.");
        state.wordTracker.status = "error";
        state.wordTracker.error = message;
        DOM.captureGuidance.textContent = message;
        setSystemMessage(`${message} Alphabet mode remains available.`, "warning");
        renderCapabilities();
        return false;
    }
}

async function startCamera() {
    if (state.cameraStarting || state.cameraRunning) return;
    primeAudioFeedback();
    state.cameraStarting = true;
    setCameraStatus("Requesting access", "checking");
    setSystemMessage("Waiting for camera permission. Video stays in this browser and recognition frames go only to this backend.");
    updateControls();

    try {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
            throw new Error("Camera access requires HTTPS or localhost in a supported browser.");
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                width: { ideal: FRAME_WIDTH },
                height: { ideal: FRAME_HEIGHT },
                facingMode: "user",
            },
        });

        state.cameraStream = stream;
        DOM.webcam.srcObject = stream;
        await waitForVideoMetadata();
        await DOM.webcam.play();

        state.cameraRunning = true;
        state.lastHolisticRequestAt = 0;
        state.fpsFrames = 0;
        DOM.cameraPlaceholder.classList.add("is-hidden");
        setCameraStatus("Running", "online");
        setSystemMessage("Camera is running. Keep your hands and upper body visible with even lighting.");
        state.cameraAnimationFrame = window.requestAnimationFrame(processCameraFrame);
        if (state.mode === "words") await prepareWordTracking();
        renderKeyboard();
    } catch (error) {
        stopMediaTracks();
        const message = friendlyCameraError(error);
        setCameraStatus(message, "error");
        setSystemMessage(message, "error");
    } finally {
        state.cameraStarting = false;
        updateControls();
    }
}

function waitForVideoMetadata() {
    if (DOM.webcam.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(() => {
            cleanup();
            reject(new Error("Camera metadata timed out."));
        }, 5_000);
        const onLoaded = () => {
            cleanup();
            resolve();
        };
        const cleanup = () => {
            window.clearTimeout(timeout);
            DOM.webcam.removeEventListener("loadedmetadata", onLoaded);
        };
        DOM.webcam.addEventListener("loadedmetadata", onLoaded, { once: true });
    });
}

function friendlyCameraError(error) {
    const name = safeText(error?.name);
    if (name === "NotAllowedError" || name === "SecurityError") return "Camera permission was denied.";
    if (name === "NotFoundError" || name === "DevicesNotFoundError") return "No camera was found.";
    if (name === "NotReadableError" || name === "TrackStartError") return "The camera is already in use or unavailable.";
    if (name === "OverconstrainedError") return "The camera cannot provide a compatible video stream.";
    return safeText(error?.message, "The camera could not start.");
}

async function processCameraFrame(timestamp) {
    if (!state.cameraRunning || state.disposed) return;
    state.cameraAnimationFrame = window.requestAnimationFrame(processCameraFrame);

    if (DOM.webcam.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    // Alphabet inference already performs hand detection on the backend. Send
    // camera frames directly so this mode does not depend on, or duplicate,
    // the browser's heavier full-body landmark tracker.
    if (state.mode === "alphabet") {
        queueAlphabetFrame(timestamp);
        return;
    }
    if (state.mode !== "words" || !state.holistic) return;
    if (state.holisticBusy || timestamp - state.lastHolisticRequestAt < HOLISTIC_INTERVAL_MS) return;

    state.holisticBusy = true;
    state.lastHolisticRequestAt = timestamp;
    try {
        await state.holistic.send({ image: DOM.webcam });
    } catch (error) {
        setSystemMessage("Hand tracking stopped unexpectedly. Stop and restart the camera.", "error");
        setCameraStatus("Tracking error", "error");
    } finally {
        state.holisticBusy = false;
    }
}

function stopMediaTracks() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach((track) => track.stop());
        state.cameraStream = null;
    }
    DOM.webcam.srcObject = null;
}

function stopCamera(announce = true) {
    if (state.cameraAnimationFrame !== null) {
        window.cancelAnimationFrame(state.cameraAnimationFrame);
        state.cameraAnimationFrame = null;
    }
    state.cameraRunning = false;
    state.holisticBusy = false;
    stopMediaTracks();
    clearOverlay();
    DOM.cameraPlaceholder.classList.remove("is-hidden");
    DOM.fpsDisplay.textContent = "0 FPS";
    state.fpsFrames = 0;
    setCameraStatus("Stopped", "offline");

    if (state.words.active || state.words.waiting) {
        abortWordCapture("Camera stopped. Reset the current sign before another capture.");
    }
    clearAlphabetRequest();
    updateControls();
    if (announce) setSystemMessage("Camera stopped. No frames are being sent.");
    renderKeyboard();
}

function clearOverlay() {
    const context = DOM.overlay.getContext("2d");
    context.clearRect(0, 0, DOM.overlay.width, DOM.overlay.height);
}

function drawCoolHand(landmarks, context, width, height, pulse) {
    if (!landmarks || !landmarks.length) return [];
    const points = landmarks.map((point) => ({ x: point.x * width, y: point.y * height }));

    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    if (typeof ALPHABET_HAND_BONES !== "undefined") {
        ALPHABET_HAND_BONES.forEach(([startIndex, endIndex, colorIndex]) => {
            const start = points[startIndex];
            const end = points[endIndex];
            if (!start || !end) return;
            const color = ALPHABET_HAND_COLORS[colorIndex];
            const gradient = context.createLinearGradient(start.x, start.y, end.x, end.y);
            gradient.addColorStop(0, "rgba(244, 251, 255, 0.82)");
            gradient.addColorStop(0.45, color);
            gradient.addColorStop(1, "rgba(244, 251, 255, 0.95)");
            context.strokeStyle = gradient;
            context.shadowColor = color;
            context.shadowBlur = 10;
            context.lineWidth = 2.6 * pulse;
            context.beginPath();
            context.moveTo(start.x, start.y);
            context.lineTo(end.x, end.y);
            context.stroke();
        });

        points.forEach((point, index) => {
            const isTip = [4, 8, 12, 16, 20].includes(index);
            const color = ALPHABET_HAND_COLORS[Math.min(4, Math.max(0, Math.floor((index - 1) / 4)))];
            context.shadowColor = isTip ? color : "rgba(117, 216, 255, 0.8)";
            context.shadowBlur = isTip ? 14 : 6;
            context.fillStyle = isTip ? color : "#f7fcff";
            context.beginPath();
            context.arc(point.x, point.y, isTip ? 4.8 * pulse : 2.8, 0, Math.PI * 2);
            context.fill();
        });
    }
    context.restore();
    return points;
}

function drawAlphabetHandOverlay(rawLandmarks, label, confidence, accepted) {
    const landmarks = normalizeOverlayLandmarks(rawLandmarks);
    const context = DOM.overlay.getContext("2d");
    context.clearRect(0, 0, DOM.overlay.width, DOM.overlay.height);
    if (!landmarks.length) return;

    const width = DOM.overlay.width;
    const height = DOM.overlay.height;
    const pulse = 0.9 + 0.1 * Math.sin(performance.now() / 180);

    const points = drawCoolHand(landmarks, context, width, height, pulse);

    const left = Math.min(...points.map((point) => point.x));
    const top = Math.min(...points.map((point) => point.y));
    const displayedLabel = normalizeDetectedLetter(label) || "HAND";
    const percentage = Math.round(clamp(confidence, 0, 1) * 100);
    const text = `${displayedLabel}  ${percentage}%`;
    context.font = "800 14px system-ui, sans-serif";
    const tagWidth = context.measureText(text).width + 18;
    const tagX = clamp(left, 8, width - tagWidth - 8);
    const tagY = clamp(top - 31, 8, height - 31);
    context.shadowBlur = 0;
    context.fillStyle = accepted ? "rgba(13, 70, 59, 0.88)" : "rgba(39, 46, 72, 0.88)";
    context.fillRect(tagX, tagY, tagWidth, 24);
    context.strokeStyle = accepted ? "rgba(84, 231, 173, 0.88)" : "rgba(117, 216, 255, 0.78)";
    context.lineWidth = 1;
    context.strokeRect(tagX + 0.5, tagY + 0.5, tagWidth - 1, 23);
    context.fillStyle = "#f7fcff";
    context.fillText(text, tagX + 9, tagY + 16.5);
    context.restore();
}

function onHolisticResults(results) {
    if (!state.cameraRunning) return;
    state.fpsFrames += 1;
    drawLandmarks(results);
    const now = performance.now();

    if (state.mode === "alphabet" && state.capabilities.alphabet.ready) {
        queueAlphabetFrame(now);
    } else if (state.mode === "words" && state.words.active) {
        sampleWordFrame(results, now);
    }
}

function drawLandmarks(results) {
    const context = DOM.overlay.getContext("2d");
    context.clearRect(0, 0, DOM.overlay.width, DOM.overlay.height);
    const width = DOM.overlay.width;
    const height = DOM.overlay.height;
    const pulse = 0.9 + 0.1 * Math.sin(performance.now() / 180);

    if (results.poseLandmarks && typeof window.drawConnectors === "function" && typeof window.drawLandmarks === "function") {
        window.drawConnectors(context, results.poseLandmarks, window.POSE_CONNECTIONS, { color: "#75d8ff", lineWidth: 2 });
        window.drawLandmarks(context, results.poseLandmarks, { color: "#f4f8fc", lineWidth: 1, radius: 1.5 });
    }
    if (results.leftHandLandmarks) {
        drawCoolHand(results.leftHandLandmarks, context, width, height, pulse);
    }
    if (results.rightHandLandmarks) {
        drawCoolHand(results.rightHandLandmarks, context, width, height, pulse);
    }
}

function queueAlphabetFrame(now) {
    if (!isWebSocketOpen() || state.alphabet.inFlight || state.alphabet.encoding) return;
    if (now - state.alphabet.lastSentAt < ALPHABET_SEND_INTERVAL_MS) return;

    state.alphabet.encoding = true;
    state.alphabet.lastSentAt = now;
    try {
        drawContainedVideo(alphabetFrameCanvas, DOM.webcam);
        const dataUrl = alphabetFrameCanvas.toDataURL("image/jpeg", 0.78);
        const image = dataUrl.slice(dataUrl.indexOf(",") + 1);
        const requestId = nextRequestId("alphabet");
        const sentAt = Date.now();
        const sent = sendWebSocket({
            type: "frame",
            mode: "alphabet",
            image,
            request_id: requestId,
            sent_at: sentAt,
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
        }, true);

        if (sent) {
            state.alphabet.inFlight = requestId;
            state.alphabet.timeout = window.setTimeout(() => {
                if (state.alphabet.inFlight !== requestId) return;
                clearAlphabetRequest(requestId);
                setSystemMessage("The alphabet request timed out; live capture will continue.", "warning");
            }, REQUEST_TIMEOUT_MS);
        }
    } catch (error) {
        setSystemMessage("The camera frame could not be encoded.", "error");
    } finally {
        state.alphabet.encoding = false;
    }
}

function drawContainedVideo(canvas, video) {
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#000000";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const sourceWidth = video.videoWidth || FRAME_WIDTH;
    const sourceHeight = video.videoHeight || FRAME_HEIGHT;
    const scale = Math.min(canvas.width / sourceWidth, canvas.height / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;
    context.drawImage(video, x, y, width, height);
}

function clearAlphabetRequest(requestId = "") {
    if (requestId && state.alphabet.inFlight && requestId !== state.alphabet.inFlight) return;
    if (state.alphabet.timeout) window.clearTimeout(state.alphabet.timeout);
    state.alphabet.timeout = null;
    state.alphabet.inFlight = null;
}

function validLandmark(landmark, useVisibility = false) {
    if (!landmark) return false;
    if (!Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) return false;
    if (useVisibility && Number.isFinite(landmark.visibility) && landmark.visibility < 0.25) return false;
    return true;
}

function averageLandmarks(first, second) {
    if (!validLandmark(first, true) || !validLandmark(second, true)) return null;
    return {
        x: (first.x + second.x) / 2,
        y: (first.y + second.y) / 2,
    };
}

function appendNormalizedPoint(target, landmark, useVisibility = false) {
    if (!validLandmark(landmark, useVisibility)) {
        target.push(-1, -1);
        return;
    }
    target.push(
        clamp(landmark.x, 0, 1) * 2 - 1,
        clamp(landmark.y, 0, 1) * 2 - 1,
    );
}

function extractWlasl55Keypoints(results) {
    const pose = Array.isArray(results.poseLandmarks) ? results.poseLandmarks : [];
    const leftHand = Array.isArray(results.leftHandLandmarks) ? results.leftHandLandmarks : [];
    const rightHand = Array.isArray(results.rightHandLandmarks) ? results.rightHandLandmarks : [];

    // WLASL Pose-TGCN retained OpenPose BODY_25 order:
    // nose, neck, right shoulder/elbow/wrist, left shoulder/elbow/wrist,
    // mid-hip, right eye, left eye, right ear, left ear.
    const body = [
        pose[0] ?? null,
        averageLandmarks(pose[11], pose[12]),
        pose[12] ?? null,
        pose[14] ?? null,
        pose[16] ?? null,
        pose[11] ?? null,
        pose[13] ?? null,
        pose[15] ?? null,
        averageLandmarks(pose[23], pose[24]),
        pose[5] ?? null,
        pose[2] ?? null,
        pose[8] ?? null,
        pose[7] ?? null,
    ];

    const values = [];
    body.forEach((landmark) => appendNormalizedPoint(values, landmark, true));
    for (let index = 0; index < 21; index += 1) appendNormalizedPoint(values, leftHand[index] ?? null);
    for (let index = 0; index < 21; index += 1) appendNormalizedPoint(values, rightHand[index] ?? null);

    if (values.length !== 110) throw new Error(`Expected 110 landmark values, generated ${values.length}`);
    return values;
}

function hasUsableHand(results) {
    return (Array.isArray(results.leftHandLandmarks) && results.leftHandLandmarks.length === 21)
        || (Array.isArray(results.rightHandLandmarks) && results.rightHandLandmarks.length === 21);
}

function startWordCapture() {
    if (DOM.startWordCapture.disabled || state.words.active || state.words.waiting) return;
    resetCurrentSign(true);

    const requestId = nextRequestId("words");
    const sentAt = Date.now();
    const sent = sendWebSocket({
        type: "capture_start",
        mode: "words",
        request_id: requestId,
        total: WORD_FRAME_TOTAL,
        sent_at: sentAt,
    }, true);

    if (!sent) {
        setSystemMessage("The word capture could not start because the stream is unavailable.", "error");
        return;
    }

    state.words.active = true;
    state.words.waiting = false;
    state.words.needsReset = false;
    state.words.captured = 0;
    state.words.requestId = requestId;
    state.words.lastSampleAt = 0;
    updateWordProgress(0, WORD_FRAME_TOTAL);
    DOM.captureGuidance.textContent = "Signing now: complete one isolated word while keeping both hands visible.";
    clearLivePrediction("Capturing word", "0 of 50 valid frames captured.");
    updateControls();
}

function sampleWordFrame(results, now) {
    if (!state.words.active || now - state.words.lastSampleAt < WORD_SAMPLE_INTERVAL_MS) return;
    if (!hasUsableHand(results)) {
        DOM.captureGuidance.textContent = "Capture paused: place at least one complete hand inside the frame.";
        return;
    }

    let landmarks;
    try {
        landmarks = extractWlasl55Keypoints(results);
    } catch (error) {
        abortWordCapture("Landmark extraction failed. Reset and try again.");
        return;
    }

    state.words.lastSampleAt = now;
    const frameIndex = state.words.captured;
    const nextFrame = frameIndex + 1;
    const sent = sendWebSocket({
        type: "frame",
        mode: "words",
        request_id: state.words.requestId,
        frame_index: frameIndex,
        total: WORD_FRAME_TOTAL,
        sent_at: Date.now(),
        landmarks,
    });

    if (!sent) {
        abortWordCapture("The stream could not keep up with the capture. Reset and try again.");
        return;
    }

    state.words.captured = nextFrame;
    updateWordProgress(nextFrame, WORD_FRAME_TOTAL);
    DOM.predictionDetail.textContent = `${nextFrame} of ${WORD_FRAME_TOTAL} valid frames captured.`;

    if (nextFrame === WORD_FRAME_TOTAL) completeWordCapture();
}

function completeWordCapture() {
    state.words.active = false;
    state.words.waiting = true;
    const sentAt = Date.now();
    sendWebSocket({
        type: "capture_end",
        mode: "words",
        request_id: state.words.requestId,
        total: WORD_FRAME_TOTAL,
        sent_at: sentAt,
    }, true);

    DOM.captureGuidance.textContent = "Capture complete. Waiting for one prediction; use Reset current sign to cancel.";
    clearLivePrediction("Analyzing word", "The 50-frame isolated sequence is being classified.");
    state.words.timeout = window.setTimeout(() => {
        if (!state.words.waiting) return;
        abortWordCapture("Word prediction timed out. Reset and try again.");
    }, WORD_RESULT_TIMEOUT_MS);
    updateControls();
}

function finishWordRequest() {
    if (state.words.timeout) window.clearTimeout(state.words.timeout);
    state.words.timeout = null;
    state.words.active = false;
    state.words.waiting = false;
    state.words.needsReset = true;
    DOM.captureGuidance.textContent = "Prediction complete. Reset the current sign before another capture.";
    updateControls();
}

function abortWordCapture(message) {
    if (state.words.timeout) window.clearTimeout(state.words.timeout);
    state.words.timeout = null;
    state.words.active = false;
    state.words.waiting = false;
    state.words.needsReset = true;
    DOM.captureGuidance.textContent = message;
    updateControls();
}

function updateWordProgress(captured, total = WORD_FRAME_TOTAL) {
    const safeTotal = Math.max(1, total);
    const safeCaptured = clamp(captured, 0, safeTotal);
    const percentage = (safeCaptured / safeTotal) * 100;
    DOM.captureProgressFill.style.width = `${percentage}%`;
    DOM.captureProgressText.textContent = `${safeCaptured} / ${safeTotal}`;
    DOM.captureProgress.setAttribute("aria-valuemax", String(safeTotal));
    DOM.captureProgress.setAttribute("aria-valuenow", String(safeCaptured));
}

function resetCurrentSign(sendReset = true) {
    clearAlphabetRequest();
    state.alphabet.armed = true;

    if (state.words.timeout) window.clearTimeout(state.words.timeout);
    state.words.timeout = null;
    state.words.active = false;
    state.words.waiting = false;
    state.words.needsReset = false;
    state.words.captured = 0;
    state.words.lastSampleAt = 0;
    state.words.requestId = null;
    updateWordProgress(0, WORD_FRAME_TOTAL);
    DOM.captureGuidance.textContent = "Hold your upper body and hands in view, then capture the sign once.";

    if (sendReset && isWebSocketOpen()) {
        sendWebSocket({
            type: "reset",
            mode: state.mode,
            request_id: nextRequestId("reset"),
        });
    }

    clearLivePrediction(
        state.mode === "words" ? "Word capture ready" : "No live prediction",
        state.mode === "words"
            ? "Capture one isolated sign when ready."
            : "Present one letter after starting the camera.",
    );
    updateControls();
    renderKeyboard();
}

function updateFps() {
    const fps = state.cameraRunning ? state.fpsFrames : 0;
    DOM.fpsDisplay.textContent = `${fps} FPS`;
    state.fpsFrames = 0;
}

function retryBackend() {
    if (state.reconnectTimer) window.clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    state.reconnectAttempt = 0;
    state.wsPhase = "idle";
    checkHealth();
    connectWebSocket();
}

function selectSuggestionIndex(index) {
    const count = state.keyboard.suggestions.length;
    if (!count) return;
    state.keyboard.selectedSuggestionIndex = ((Math.round(index) % count) + count) % count;
    const buttons = [...DOM.wordSuggestions.querySelectorAll(".suggestion-chip")];
    buttons.forEach((button, buttonIndex) => {
        const selected = buttonIndex === state.keyboard.selectedSuggestionIndex;
        button.classList.toggle("is-selected", selected);
        button.setAttribute("aria-selected", String(selected));
    });
    const suggestion = selectedSuggestion();
    DOM.keyboardConfirm.textContent = suggestion ? `Accept “${suggestion}”` : "Accept best suggestion";
}

function isEditableKeyboardTarget(target) {
    return target instanceof Element && Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function handleSignKeyboardKeydown(event) {
    if (state.activePage !== "sign-input" || event.defaultPrevented || event.isComposing
            || event.ctrlKey || event.metaKey || event.altKey) return;

    const suggestionButton = event.target instanceof Element ? event.target.closest(".suggestion-chip") : null;
    if (suggestionButton && event.key === "Enter") {
        event.preventDefault();
        if (!event.repeat) commitKeyboardWord(suggestionButton.dataset.word);
        return;
    }
    if (isEditableKeyboardTarget(event.target)) return;
    if (event.target instanceof Element && event.target.closest("a, summary")) return;
    if (event.target instanceof Element && event.target.closest("button")) {
        const isKeyboardControl = Boolean(event.target.closest(
            "#keyboard-backspace, #keyboard-clear-letters, #keyboard-space, #keyboard-confirm, #keyboard-clear-sentence, #sound-toggle",
        ));
        if (!isKeyboardControl || ["Enter", " "].includes(event.key)) return;
    }

    if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(event.key)
            && state.keyboard.suggestions.length) {
        event.preventDefault();
        const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
        selectSuggestionIndex(state.keyboard.selectedSuggestionIndex + direction);
        return;
    }
    if (event.key === "Enter") {
        event.preventDefault();
        if (event.repeat) return;
        if (state.keyboard.pendingLetterAcceptable && !state.keyboard.pendingLetterCommitted) {
            acceptPendingDetectedLetter();
            return;
        }
        const suggestion = selectedSuggestion();
        if (suggestion) commitKeyboardWord(suggestion);
        else commitKeyboardWord();
        return;
    }
    if (event.key === "Backspace" && state.keyboard.spelling) {
        event.preventDefault();
        backspaceCurrentLetters();
    }
}

function bindEvents() {
    DOM.retryBackend.addEventListener("click", retryBackend);
    DOM.startCamera.addEventListener("click", startCamera);
    DOM.stopCamera.addEventListener("click", () => stopCamera(true));
    DOM.modeAlphabet.addEventListener("click", () => setMode("alphabet"));
    DOM.modeWords.addEventListener("click", () => setMode("words"));
    DOM.startWordCapture.addEventListener("click", startWordCapture);
    DOM.resetSign.addEventListener("click", () => resetCurrentSign(true));
    DOM.clearHistory.addEventListener("click", () => {
        state.history = [];
        renderHistory();
    });
    DOM.navSignInput.addEventListener("click", () => setActivePage("sign-input"));
    DOM.navTextToSign.addEventListener("click", () => setActivePage("text-to-sign"));
    window.addEventListener("hashchange", () => setActivePage(pageFromHash(), false));

    DOM.soundToggle.addEventListener("click", toggleSound);
    DOM.wordSuggestions.addEventListener("click", (event) => {
        const button = event.target.closest(".suggestion-chip");
        if (button && DOM.wordSuggestions.contains(button)) {
            selectSuggestionIndex(Number(button.dataset.suggestionIndex));
            commitKeyboardWord(button.dataset.word);
        }
    });
    DOM.wordSuggestions.addEventListener("pointerover", (event) => {
        const button = event.target.closest(".suggestion-chip");
        if (button && DOM.wordSuggestions.contains(button)) {
            selectSuggestionIndex(Number(button.dataset.suggestionIndex));
        }
    });
    DOM.keyboardBackspace.addEventListener("click", backspaceCurrentLetters);
    DOM.keyboardClearLetters.addEventListener("click", clearCurrentLetters);
    DOM.keyboardSpace.addEventListener("click", () => commitKeyboardWord());
    DOM.keyboardConfirm.addEventListener("click", () => commitKeyboardWord(selectedSuggestion()));
    DOM.keyboardClearSentence.addEventListener("click", clearSentence);

    DOM.textSignForm.addEventListener("submit", (event) => {
        event.preventDefault();
        translateTypedText();
    });
    DOM.translationInput.addEventListener("input", updateTranslationCharacterCount);
    DOM.clearTranslationInput.addEventListener("click", clearTranslationInput);
    DOM.startRecording.addEventListener("click", startRecording);
    DOM.stopRecording.addEventListener("click", () => stopRecording(true));
    DOM.previousSign.addEventListener("click", () => moveSequence(-1));
    DOM.nextSign.addEventListener("click", () => moveSequence(1));
    DOM.sequenceAction.addEventListener("click", activateSequenceAction);
    DOM.signSequence.addEventListener("click", (event) => {
        const item = event.target.closest(".sign-sequence-item");
        if (item && DOM.signSequence.contains(item)) setActiveSequenceIndex(Number(item.dataset.sequenceIndex));
    });
    document.addEventListener("keydown", handleSignKeyboardKeydown);
    window.addEventListener("pagehide", cleanup, { once: true });
}

function cleanup() {
    if (state.disposed) return;
    state.disposed = true;
    if (state.reconnectTimer) window.clearTimeout(state.reconnectTimer);
    if (state.fpsTimer) window.clearInterval(state.fpsTimer);
    if (state.words.timeout) window.clearTimeout(state.words.timeout);
    if (state.keyboard.cooldownTimer) window.clearTimeout(state.keyboard.cooldownTimer);
    cancelSuggestionRequest();
    clearAlphabetRequest();
    stopCamera(false);
    if (isVoiceInputActive()) stopRecording(false);
    stopMicrophoneTracks();
    clearRecordingTimers();

    if (state.websocket) {
        try {
            state.websocket.close(1000, "Page closed");
        } catch (error) {
            // The socket may already be closing.
        }
        state.websocket = null;
    }

    if (state.holistic && typeof state.holistic.close === "function") {
        try {
            state.holistic.close();
        } catch (error) {
            // MediaPipe cleanup is best-effort during page shutdown.
        }
    }
    state.holistic = null;
    if (state.keyboard.audioContext && typeof state.keyboard.audioContext.close === "function") {
        state.keyboard.audioContext.close().catch(() => {});
    }
    state.keyboard.audioContext = null;
}

function initializeApp() {
    cacheDom();
    bindEvents();
    renderHistory();
    renderTopPredictions([]);
    renderKeyboard();
    renderMode();
    renderCapabilities();
    updateWordProgress(0, WORD_FRAME_TOTAL);
    setConfidence(0);
    setCameraStatus("Stopped", "offline");
    updateTranslationCharacterCount();
    setActiveSequenceIndex(0);
    setActivePage(pageFromHash(), false);
    state.fpsTimer = window.setInterval(updateFps, 1_000);
    checkHealth();
    connectWebSocket();
    fetchVocabulary();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeApp, { once: true });
} else {
    initializeApp();
}
