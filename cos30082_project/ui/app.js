/**
 * Face Recognition Attendance System — Frontend Controller (LIVE BACKEND)
 * Captures real webcam frames, sends them to the Colab Flask backend,
 * and drives the existing index.html UI (face box, liveness badge,
 * registration, redirect to userfound.html).
 */

// ════════════════════════════════════════════════════════════════════════════
//  >>> PASTE YOUR NGROK URL FROM THE COLAB CELL HERE <<<
//  e.g. const BACKEND_URL = "https://abcd-12-34-56.ngrok-free.app";
// ════════════════════════════════════════════════════════════════════════════
const BACKEND_URL = "http://localhost:5001";

// How often to send a frame to the backend (ms). 1500 = every 1.5s.
const SCAN_INTERVAL = 1500;

// ── DOM handles ───────────────────────────────────────────────────────────────
let webcamStream = null;
let videoElement = null;
let scanTimer = null;
let isBusy = false;
let lastRegisterFrame = null; // holds latest frame for registration capture

const videoContainer = document.querySelector('.relative.aspect-video');

// ── 1. Webcam init ────────────────────────────────────────────────────────────
async function initializeHardwareWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false
        });

        const placeholder = document.getElementById('webcam-placeholder');
        if (placeholder) placeholder.classList.add('hidden');

        videoElement = document.createElement('video');
        videoElement.id = 'webcam-feed';
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;
        videoElement.className = 'w-full h-full object-cover rounded-lg';
        videoElement.srcObject = webcamStream;
        videoContainer.appendChild(videoElement);

        videoElement.onloadedmetadata = () => {
            // Begin the scanning loop once the camera is streaming
            startScanLoop();
        };
    } catch (err) {
        console.error("Camera access failed:", err);
        const textPlaceholder = document.querySelector('#webcam-placeholder p');
        if (textPlaceholder) {
            textPlaceholder.innerText = "Camera Access Blocked / Missing Permissions";
            textPlaceholder.classList.add('text-rose-500', 'font-bold');
        }
    }
}

// ── 2. Capture current frame as base64 JPEG ───────────────────────────────────
function captureFrame() {
    if (!videoElement || videoElement.videoWidth === 0) return null;
    const c = document.createElement('canvas');
    c.width = videoElement.videoWidth;
    c.height = videoElement.videoHeight;
    c.getContext('2d').drawImage(videoElement, 0, 0, c.width, c.height);
    return c.toDataURL('image/jpeg', 0.8);
}

// ── 3. Scan loop: send a frame to backend every SCAN_INTERVAL ms ──────────────
function startScanLoop() {
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = setInterval(async () => {
        if (isBusy || BACKEND_URL.includes("PASTE")) return;
        const frame = captureFrame();
        if (!frame) return;
        lastRegisterFrame = frame;
        isBusy = true;
        try {
            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frame })
            });
            const data = await res.json();
            applyBackendResult(data);
        } catch (e) {
            console.error("Backend call failed:", e);
            setStatusText("Backend unreachable — check ngrok URL", "error");
        } finally {
            isBusy = false;
        }
    }, SCAN_INTERVAL);
}

// ── 4. Drive the existing UI based on backend result ──────────────────────────
function applyBackendResult(data) {
    const idleNotice    = document.getElementById('idle-notice');
    const insightsPanel = document.getElementById('insights-panel');
    const faceBox       = document.getElementById('face-box');
    const faceBoxLabel  = document.getElementById('face-box-label');
    const cardAlert     = document.getElementById('card-alert');
    const txtAlertTitle = document.getElementById('txt-alert-title');
    const txtAlertStatus= document.getElementById('txt-alert-status');
    const txtAlertDetails = document.getElementById('txt-alert-details');
    const btnInlineReg  = document.getElementById('btn-inline-register');
    const badgeLiveness = document.getElementById('badge-liveness');
    const txtLivenessDesc = document.getElementById('txt-liveness-desc');

    // Reveal panels, hide idle
    idleNotice.classList.add('hidden');
    insightsPanel.classList.remove('hidden');
    faceBox.classList.remove('hidden');
    btnInlineReg.classList.add('hidden');

    // Liveness badge
    if (data.liveness === "REAL") {
        badgeLiveness.className = "font-black tracking-wide text-sm px-3 py-1.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800";
        badgeLiveness.innerText = `REAL (${data.liveness_conf}%)`;
        txtLivenessDesc.innerText = "Liveness signature verified against hardware templates.";
    } else {
        badgeLiveness.className = "font-black tracking-wide text-sm px-3 py-1.5 rounded bg-rose-600 text-white border border-rose-500 animate-pulse-red";
        badgeLiveness.innerText = `SPOOF (${data.liveness_conf}%)`;
        txtLivenessDesc.innerText = "Spoof detected (printed photo / screen image).";
    }

    if (data.state === 'SPOOF') {
        faceBox.className = "absolute border-2 border-rose-500 rounded-md transition-all duration-300";
        faceBoxLabel.className = "absolute -top-6 left-0 bg-rose-500 text-white text-xs font-bold px-1.5 py-0.5 rounded shadow";
        faceBoxLabel.innerText = data.criminal_flag ? "WATCHLIST MATCH" : "CRITICAL THREAT";
        cardAlert.className = "p-5 rounded-xl border border-rose-500/40 bg-gray-800 shadow-lg shadow-rose-950/20";
        txtAlertTitle.className = "text-xs font-semibold uppercase tracking-wider text-rose-400 mb-1";
        txtAlertTitle.innerText = data.criminal_flag ? "Watchlist Alert" : "Security Intercept";
        txtAlertStatus.innerText = data.criminal_flag ? data.name : "Access Warning";
        txtAlertDetails.innerText = data.criminal_flag
            ? "This individual matches a flagged watchlist entry."
            : "Biometric match flagged due to invalid structural credentials.";
        return;
    }

    if (data.state === 'SUCCESS') {
        // Stop scanning and redirect to the verified page
        clearInterval(scanTimer);
        const session = {
            name: data.name,
            employee_id: data.employee_id,
            timestamp: new Date().toLocaleTimeString(),
            emotion_label: data.emotion_label,
            emotion_icon: data.emotion_icon
        };
        sessionStorage.setItem("recognizedUser", JSON.stringify(session));
        window.location.href = "userfound.html";
        return;
    }

    // UNKNOWN — real face, not in database
    faceBox.className = "absolute border-2 border-amber-500 rounded-md transition-all duration-300";
    faceBoxLabel.className = "absolute -top-6 left-0 bg-amber-500 text-gray-900 text-xs font-bold px-1.5 py-0.5 rounded shadow";
    faceBoxLabel.innerText = "UNREGISTERED FACE";
    cardAlert.className = "p-5 rounded-xl border border-amber-500/40 bg-gray-800 shadow-lg shadow-amber-950/20";
    txtAlertTitle.className = "text-xs font-semibold uppercase tracking-wider text-amber-400 mb-1";
    txtAlertTitle.innerText = "Database Mismatch";
    txtAlertStatus.innerText = "User Not Found";
    txtAlertDetails.innerText = `Unrecognised face (best match ${data.identity_conf}%). Emotion: ${data.emotion_icon} ${data.emotion_label}. Register to add a new profile.`;
    btnInlineReg.classList.remove('hidden');
}

function setStatusText(msg, kind) {
    const txtAlertStatus = document.getElementById('txt-alert-status');
    const idleNotice = document.getElementById('idle-notice');
    const insightsPanel = document.getElementById('insights-panel');
    if (idleNotice) idleNotice.classList.add('hidden');
    if (insightsPanel) insightsPanel.classList.remove('hidden');
    if (txtAlertStatus) txtAlertStatus.innerText = msg;
}

// ── 5. Registration: capture current face and POST to backend ─────────────────
async function submitRegistrationLive() {
    const name = document.getElementById('reg-name').value || "Jane Doe";
    const id   = document.getElementById('reg-id').value || "EMP-8392";
    const frame = lastRegisterFrame || captureFrame();
    if (!frame || BACKEND_URL.includes("PASTE")) {
        alert("No frame or backend URL not set.");
        return;
    }
    try {
        const res = await fetch(`${BACKEND_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frame, name: name, employee_id: id })
        });
        const data = await res.json();
        if (data.ok) {
            // Save session and go to the confirmation page
            const session = {
                name: name, employee_id: id,
                timestamp: new Date().toLocaleTimeString(),
                emotion_label: "Neutral", emotion_icon: "😐"
            };
            sessionStorage.setItem("recognizedUser", JSON.stringify(session));
            window.location.href = "userfound.html";
        }
    } catch (e) {
        console.error("Registration failed:", e);
        alert("Registration failed — backend unreachable.");
    }
}

// Expose registration drawer controls (used by inline onclick handlers)
function triggerRegistrationForm() {
    document.getElementById('registration-drawer').classList.remove('pointer-events-none', 'opacity-0');
    document.getElementById('registration-drawer').firstElementChild.classList.remove('translate-x-full');
}
function closeRegistrationDrawer() {
    document.getElementById('registration-drawer').classList.add('pointer-events-none', 'opacity-0');
    document.getElementById('registration-drawer').firstElementChild.classList.add('translate-x-full');
}

// ── Employee management ───────────────────────────────────────────────────────
async function openManageDrawer() {
    const drawer = document.getElementById('manage-drawer');
    drawer.classList.remove('pointer-events-none', 'opacity-0');
    drawer.firstElementChild.classList.remove('translate-x-full');
    await refreshEmployeeList();
}

function closeManageDrawer() {
    const drawer = document.getElementById('manage-drawer');
    drawer.classList.add('pointer-events-none', 'opacity-0');
    drawer.firstElementChild.classList.add('translate-x-full');
}

async function refreshEmployeeList() {
    const listEl = document.getElementById('employee-list');
    try {
        const res = await fetch(`${BACKEND_URL}/health`);
        const data = await res.json();
        const names = data.employees || [];
        if (names.length === 0) {
            listEl.innerHTML = '<p class="text-gray-500 text-sm">No employees registered yet.</p>';
            return;
        }
        listEl.innerHTML = names.map(name => `
            <div class="flex justify-between items-center bg-gray-900 border border-gray-700 rounded-lg px-4 py-3">
                <span class="text-white font-medium">${name}</span>
                <button onclick="deleteEmployee('${name.replace(/'/g, "\\'")}')"
                    class="bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold px-3 py-1.5 rounded-md transition">
                    Delete
                </button>
            </div>
        `).join('');
    } catch (e) {
        listEl.innerHTML = '<p class="text-rose-500 text-sm">Backend unreachable.</p>';
    }
}

async function deleteEmployee(name) {
    if (!confirm(`Delete ${name}? This cannot be undone.`)) return;
    try {
        const res = await fetch(`${BACKEND_URL}/delete_employee`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });
        const data = await res.json();
        if (data.ok) {
            await refreshEmployeeList();   // refresh the list
        } else {
            alert('Delete failed: ' + (data.error || 'unknown'));
        }
    } catch (e) {
        alert('Delete failed — backend unreachable.');
    }
}

// ── 6. Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initializeHardwareWebcam();
    // Make functions available to the inline HTML onclick attributes
    window.triggerRegistrationForm = triggerRegistrationForm;
    window.closeRegistrationDrawer = closeRegistrationDrawer;
    window.submitRegistrationLive  = submitRegistrationLive;
    window.openManageDrawer        = openManageDrawer;  
    window.closeManageDrawer       = closeManageDrawer;  
    window.deleteEmployee          = deleteEmployee;     
});
