/**
 * VisionAttend - AI Face Recognition Attendance Portal
 * Frontend Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Application State ---
    let state = {
        activeTab: 'tab-scanner',
        isCameraActive: true,
        scanIntervalId: null,
        isProcessingFrame: false,
        lastRecognizedMap: {}, // student_id -> last_timestamp
        students: [],
        attendance: [],
        stats: {}
    };

    // --- DOM Elements ---
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const systemClock = document.getElementById('system-clock');
    
    // Camera Elements
    const videoElem = document.getElementById('webcam-video');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const cameraIcon = document.getElementById('camera-icon');
    const cameraBtnText = document.getElementById('camera-btn-text');
    const scannerStatus = document.getElementById('scanner-status');
    const recognitionFeed = document.getElementById('recognition-feed');

    // Mode toggles
    const btnModeWebcam = document.getElementById('mode-webcam');
    const btnModeUpload = document.getElementById('mode-upload');
    const webcamContainer = document.getElementById('webcam-container');
    const uploadContainer = document.getElementById('upload-container');
    const uploadInput = document.getElementById('upload-input');
    const imageDropzone = document.getElementById('image-dropzone');
    const uploadPreviewWrapper = document.getElementById('upload-preview-wrapper');
    const uploadPreviewImg = document.getElementById('upload-preview-img');
    const uploadCanvas = document.getElementById('upload-canvas');

    // Filter elements
    const filterSearch = document.getElementById('filter-search');
    const filterDate = document.getElementById('filter-date');
    const filterStatus = document.getElementById('filter-status');
    const attendanceTbody = document.getElementById('attendance-tbody');
    const btnExportCsv = document.getElementById('btn-export-csv');

    // Register Student Modal Elements
    const registerModal = document.getElementById('register-modal');
    const btnAddStudentModal = document.getElementById('btn-add-student-modal');
    const btnQuickRegister = document.getElementById('btn-quick-register');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnCancelModal = document.getElementById('btn-cancel-modal');
    const registerForm = document.getElementById('register-form');
    const snapVideo = document.getElementById('snap-video');
    const snapCanvas = document.getElementById('snap-canvas');
    const btnCaptureSnapshot = document.getElementById('btn-capture-snapshot');
    const regPreviewBox = document.getElementById('reg-preview-box');
    const regPreviewImg = document.getElementById('reg-preview-img');
    const snapModeCamera = document.getElementById('snap-mode-camera');
    const snapModeFile = document.getElementById('snap-mode-file');
    const snapBoxCamera = document.getElementById('snap-box-camera');
    const snapBoxFile = document.getElementById('snap-box-file');
    const regFileInput = document.getElementById('reg-file-input');

    // Model Retrain
    const btnTriggerTrain = document.getElementById('btn-trigger-train');
    const btnFullRetrain = document.getElementById('btn-full-retrain');

    // --- Audio Feedback (Web Audio API Synthesizer) ---
    function playChime(type = 'success') {
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);

            if (type === 'success') {
                osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
                osc.frequency.setValueAtTime(880, ctx.currentTime + 0.1); // A5
                gain.gain.setValueAtTime(0.15, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.35);
            } else {
                osc.frequency.setValueAtTime(300, ctx.currentTime);
                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.25);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.25);
            }
        } catch (e) {
            console.log('Audio disabled/not supported');
        }
    }

    // --- Toast Notifications ---
    function showToast(message, type = 'success') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let iconClass = 'fa-check-circle';
        if (type === 'warning') iconClass = 'fa-exclamation-triangle';
        if (type === 'error') iconClass = 'fa-times-circle';

        toast.innerHTML = `
            <i class="fa-solid ${iconClass}"></i>
            <span>${message}</span>
        `;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // --- Clock Update ---
    function updateClock() {
        const now = new Date();
        systemClock.textContent = now.toLocaleTimeString() + ' | ' + now.toLocaleDateString();
    }
    setInterval(updateClock, 1000);
    updateClock();

    // --- Navigation & Tabs ---
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            state.activeTab = targetTab;

            // Page title update
            const titles = {
                'tab-scanner': ['Live AI Scanner', 'Position your face in front of the camera to automatically mark attendance.'],
                'tab-attendance': ['Attendance Records', 'Search, filter, and export daily attendance logs.'],
                'tab-students': ['Student Directory', 'Manage classmate profiles and enrolled face images.'],
                'tab-analytics': ['Analytics & Model Diagnostics', 'View attendance metrics and manage AI face recognition model.']
            };
            document.getElementById('page-title').textContent = titles[targetTab][0];
            document.getElementById('page-desc').textContent = titles[targetTab][1];

            // Refresh tab specific data
            if (targetTab === 'tab-attendance') loadAttendanceTable();
            if (targetTab === 'tab-students') loadStudentGrid();
            if (targetTab === 'tab-analytics') loadAnalyticsTab();
        });
    });

    // --- Webcam Initialization ---
    async function initWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
            });
            videoElem.srcObject = stream;
            videoElem.onloadedmetadata = () => {
                videoElem.play();
                startScannerLoop();
            };
        } catch (err) {
            console.error("Camera access error:", err);
            showToast("Camera access unavailable. You can use Photo Upload mode.", "warning");
            scannerStatus.innerHTML = `<span class="status-dot red"></span><span>Camera Offline</span>`;
        }
    }

    function toggleCamera() {
        state.isCameraActive = !state.isCameraActive;
        if (state.isCameraActive) {
            if (videoElem.srcObject) {
                videoElem.play();
            } else {
                initWebcam();
            }
            cameraIcon.className = "fa-solid fa-video-slash";
            cameraBtnText.textContent = "Pause Camera";
            scannerStatus.innerHTML = `<span class="status-pulse active"></span><span>Scanning active...</span>`;
        } else {
            videoElem.pause();
            cameraIcon.className = "fa-solid fa-video";
            cameraBtnText.textContent = "Resume Camera";
            scannerStatus.innerHTML = `<span class="status-dot red"></span><span>Camera Paused</span>`;
            clearCanvas(overlayCanvas);
        }
    }

    btnToggleCamera.addEventListener('click', toggleCamera);

    // --- Mode Switch (Webcam vs Photo Upload) ---
    btnModeWebcam.addEventListener('click', () => {
        btnModeWebcam.classList.add('active');
        btnModeUpload.classList.remove('active');
        webcamContainer.classList.remove('hidden');
        uploadContainer.classList.add('hidden');
        state.isCameraActive = true;
        startScannerLoop();
    });

    btnModeUpload.addEventListener('click', () => {
        btnModeUpload.classList.add('active');
        btnModeWebcam.classList.remove('active');
        uploadContainer.classList.remove('hidden');
        webcamContainer.classList.add('hidden');
        state.isCameraActive = false;
        clearCanvas(overlayCanvas);
    });

    // Image Upload Handlers
    imageDropzone.addEventListener('click', () => uploadInput.click());
    
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) processUploadedFile(file);
    });

    imageDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = 'var(--cyan)';
    });

    imageDropzone.addEventListener('dragleave', () => {
        imageDropzone.style.borderColor = 'rgba(99, 102, 241, 0.4)';
    });

    imageDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) processUploadedFile(file);
    });

    function processUploadedFile(file) {
        const reader = new FileReader();
        reader.onload = function(evt) {
            const b64Image = evt.target.result;
            uploadPreviewImg.src = b64Image;
            uploadPreviewWrapper.classList.remove('hidden');
            imageDropzone.classList.add('hidden');

            uploadPreviewImg.onload = () => {
                sendFrameForRecognition(b64Image, uploadCanvas, uploadPreviewImg);
            };
        };
        reader.readAsDataURL(file);
    }

    // --- Real-time Face Scanner Loop ---
    function startScannerLoop() {
        if (state.scanIntervalId) clearInterval(state.scanIntervalId);
        state.scanIntervalId = setInterval(() => {
            if (state.isCameraActive && state.activeTab === 'tab-scanner' && !state.isProcessingFrame) {
                captureAndRecognizeFrame();
            }
        }, 1200); // Frame scan rate: ~1.2s
    }

    function captureAndRecognizeFrame() {
        if (!videoElem.videoWidth || videoElem.paused) return;

        state.isProcessingFrame = true;
        const canvas = document.createElement('canvas');
        canvas.width = videoElem.videoWidth;
        canvas.height = videoElem.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);

        const frameB64 = canvas.toDataURL('image/jpeg', 0.8);
        sendFrameForRecognition(frameB64, overlayCanvas, videoElem);
    }

    async function sendFrameForRecognition(b64Data, targetCanvas, mediaSourceElem) {
        try {
            const response = await fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: b64Data })
            });
            const data = await response.json();

            if (data.status === 'success') {
                drawBoundingBoxes(data.results, targetCanvas, mediaSourceElem);
                handleRecognitionResults(data.results);
            } else {
                clearCanvas(targetCanvas);
            }
        } catch (err) {
            console.error("Recognition API error:", err);
        } finally {
            state.isProcessingFrame = false;
        }
    }

    // --- Canvas Overlay Bounding Boxes ---
    function clearCanvas(canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function drawBoundingBoxes(results, canvas, mediaElem) {
        const displayWidth = mediaElem.clientWidth || mediaElem.videoWidth || 640;
        const displayHeight = mediaElem.clientHeight || mediaElem.videoHeight || 360;

        const naturalWidth = mediaElem.videoWidth || mediaElem.naturalWidth || displayWidth;
        const naturalHeight = mediaElem.videoHeight || mediaElem.naturalHeight || displayHeight;

        canvas.width = displayWidth;
        canvas.height = displayHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, displayWidth, displayHeight);

        const scaleX = displayWidth / naturalWidth;
        const scaleY = displayHeight / naturalHeight;

        results.forEach(res => {
            const [top, right, bottom, left] = res.box;

            const x = left * scaleX;
            const y = top * scaleY;
            const width = (right - left) * scaleX;
            const height = (bottom - top) * scaleY;

            const isRecognized = res.recognized;
            const strokeColor = isRecognized ? '#10b981' : '#ef4444';
            const labelText = isRecognized ? `${res.student.name} (${res.confidence}%)` : `Unknown Face (${res.confidence}%)`;

            // Draw bounding box
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 3;
            ctx.shadowColor = strokeColor;
            ctx.shadowBlur = 12;
            ctx.strokeRect(x, y, width, height);

            // Draw corner accents
            ctx.fillStyle = strokeColor;
            const cornerSize = 8;
            ctx.fillRect(x - 2, y - 2, cornerSize, cornerSize);
            ctx.fillRect(x + width - cornerSize + 2, y - 2, cornerSize, cornerSize);
            ctx.fillRect(x - 2, y + height - cornerSize + 2, cornerSize, cornerSize);
            ctx.fillRect(x + width - cornerSize + 2, y + height - cornerSize + 2, cornerSize, cornerSize);

            // Draw top label pill
            ctx.font = '600 13px Outfit, sans-serif';
            const textWidth = ctx.measureText(labelText).width;
            const pillHeight = 26;

            ctx.shadowBlur = 0;
            ctx.fillStyle = isRecognized ? 'rgba(16, 185, 129, 0.9)' : 'rgba(239, 68, 68, 0.9)';
            ctx.beginPath();
            ctx.roundRect(x, Math.max(0, y - pillHeight - 6), textWidth + 16, pillHeight, 6);
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, x + 8, Math.max(16, y - 12));
        });
    }

    // --- Feed & Notification Updates ---
    function handleRecognitionResults(results) {
        let refreshStatsNeeded = false;

        results.forEach(res => {
            if (res.recognized) {
                const s = res.student;
                const action = res.action;

                if (action === 'MARKED_PRESENT' || action === 'MARKED_LATE') {
                    playChime('success');
                    showToast(`Attendance marked for ${s.name} (${action === 'MARKED_PRESENT' ? 'Present' : 'Late'})!`, 'success');
                    addToFeed(s, res.confidence, res.marked_time, action);
                    refreshStatsNeeded = true;
                } else if (action === 'ALREADY_MARKED') {
                    // Only add to feed once per 10 seconds to avoid spamming
                    const now = Date.now();
                    if (!state.lastRecognizedMap[s.student_id] || (now - state.lastRecognizedMap[s.student_id]) > 10000) {
                        state.lastRecognizedMap[s.student_id] = now;
                        addToFeed(s, res.confidence, res.marked_time, 'ALREADY_MARKED');
                    }
                }
            }
        });

        if (refreshStatsNeeded) {
            loadStats();
        }
    }

    function addToFeed(student, confidence, timeStr, action) {
        // Remove feed placeholder if present
        const placeholder = recognitionFeed.querySelector('.feed-placeholder');
        if (placeholder) placeholder.remove();

        const feedItem = document.createElement('div');
        feedItem.className = 'feed-item';

        const badgeClass = action === 'ALREADY_MARKED' ? 'already' : 'present';
        const badgeLabel = action === 'ALREADY_MARKED' ? 'Recorded' : 'Verified';

        feedItem.innerHTML = `
            <img src="${student.profile_pic}" class="feed-avatar" alt="${student.name}" onerror="this.src='/static/profiles/default.png'">
            <div class="feed-details">
                <h4>${student.name}</h4>
                <p>${student.roll_no} &bull; ${timeStr}</p>
            </div>
            <span class="feed-badge ${badgeClass}">${badgeLabel}</span>
        `;

        recognitionFeed.insertBefore(feedItem, recognitionFeed.firstChild);

        // Keep last 10 entries
        while (recognitionFeed.children.length > 10) {
            recognitionFeed.removeChild(recognitionFeed.lastChild);
        }
    }

    // --- API & Data Loading ---
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.status === 'success') {
                state.stats = data;
                document.getElementById('stat-total-students').textContent = data.total_students;
                document.getElementById('stat-present-today').textContent = data.present_today;
                document.getElementById('stat-absent-today').textContent = data.absent_today;
                document.getElementById('stat-attendance-rate').textContent = `${data.attendance_rate}%`;

                // Update model status chip
                const modelChip = document.getElementById('chip-model-status');
                if (data.model_status.is_trained) {
                    modelChip.innerHTML = `<span class="status-dot green"></span><span class="chip-text">AI Model Ready (${data.model_status.total_encodings} Encodings)</span>`;
                } else {
                    modelChip.innerHTML = `<span class="status-dot red"></span><span class="chip-text">Model Not Trained</span>`;
                }
            }
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    }

    async function loadAttendanceTable() {
        try {
            const query = filterSearch.value;
            const date = filterDate.value;
            const status = filterStatus.value;

            const url = `/api/attendance?date=${encodeURIComponent(date)}&status=${encodeURIComponent(status)}&query=${encodeURIComponent(query)}`;
            const res = await fetch(url);
            const data = await res.json();

            if (data.status === 'success') {
                state.attendance = data.attendance;
                renderAttendanceRows(data.attendance);
            }
        } catch (err) {
            console.error("Failed to load attendance:", err);
        }
    }

    function renderAttendanceRows(rows) {
        attendanceTbody.innerHTML = '';
        if (rows.length === 0) {
            attendanceTbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 30px;">No attendance records found matching filters.</td></tr>`;
            return;
        }

        rows.forEach(row => {
            const tr = document.createElement('tr');
            const statusClass = row.status.toLowerCase();
            tr.innerHTML = `
                <td><strong>${row.name}</strong></td>
                <td><span style="color: var(--cyan); font-weight: 600;">${row.roll_no}</span></td>
                <td>${row.department}</td>
                <td>${row.date}</td>
                <td>${row.time}</td>
                <td><span class="status-pill ${statusClass}">${row.status}</span></td>
                <td>${row.confidence}%</td>
                <td><small style="color: var(--text-muted);">${row.source}</small></td>
                <td>
                    <button class="btn btn-outline btn-toggle-status" data-id="${row.student_id}" data-status="${row.status}">
                        Toggle Status
                    </button>
                </td>
            `;
            attendanceTbody.appendChild(tr);
        });

        // Add status toggle handlers
        document.querySelectorAll('.btn-toggle-status').forEach(btn => {
            btn.addEventListener('click', async () => {
                const sId = btn.getAttribute('data-id');
                const currStatus = btn.getAttribute('data-status');
                const newStatus = currStatus === 'Present' ? 'Absent' : 'Present';

                await fetch('/api/attendance/manual', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ student_id: sId, status: newStatus })
                });
                showToast(`Status updated to ${newStatus}`, 'success');
                loadAttendanceTable();
                loadStats();
            });
        });
    }

    // Filter Listeners
    filterSearch.addEventListener('input', loadAttendanceTable);
    filterDate.addEventListener('change', loadAttendanceTable);
    filterStatus.addEventListener('change', loadAttendanceTable);

    // CSV Export
    btnExportCsv.addEventListener('click', () => {
        window.location.href = '/api/export';
        showToast('Exporting CSV log file...', 'success');
    });

    // Student Directory
    async function loadStudentGrid() {
        try {
            const res = await fetch('/api/students');
            const data = await res.json();
            if (data.status === 'success') {
                state.students = data.students;
                renderStudentGrid(data.students);
            }
        } catch (err) {
            console.error("Failed to load students:", err);
        }
    }

    function renderStudentGrid(students) {
        const grid = document.getElementById('student-grid');
        grid.innerHTML = '';
        if (students.length === 0) {
            grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 40px;">No registered students found. Register your classmates now!</div>`;
            return;
        }

        students.forEach(s => {
            const card = document.createElement('div');
            card.className = 'student-card';
            card.innerHTML = `
                <img src="${s.profile_pic}" class="student-avatar" alt="${s.name}" onerror="this.src='/static/profiles/default.png'">
                <h3 class="student-name">${s.name}</h3>
                <div class="student-roll">${s.roll_no}</div>
                <div class="student-dept">${s.department}</div>
                <div style="font-size: 11px; color: var(--text-muted);">Enrolled: ${s.registered_at}</div>
            `;
            grid.appendChild(card);
        });
    }

    // Analytics Tab
    async function loadAnalyticsTab() {
        await loadStats();
        const data = state.stats;

        // Department Breakdown
        const deptList = document.getElementById('dept-stats-list');
        deptList.innerHTML = '';
        const counts = data.department_counts || {};
        const total = data.total_students || 1;

        for (const [dept, count] of Object.entries(counts)) {
            const pct = Math.round((count / total) * 100);
            const item = document.createElement('div');
            item.className = 'dept-item';
            item.innerHTML = `
                <div class="dept-info">
                    <span>${dept}</span>
                    <span><strong>${count}</strong> students (${pct}%)</span>
                </div>
                <div class="dept-bar-bg">
                    <div class="dept-bar-fill" style="width: ${pct}%"></div>
                </div>
            `;
            deptList.appendChild(item);
        }

        // Model diagnostics
        const m = data.model_status || {};
        document.getElementById('model-total-encodings').textContent = m.total_encodings || 0;
        document.getElementById('model-last-trained').textContent = m.trained_at || 'N/A';
    }

    // --- Modal Register Student ---
    function openModal() {
        registerModal.classList.remove('hidden');
        initSnapCamera();
    }

    function closeModal() {
        registerModal.classList.add('hidden');
        if (snapVideo.srcObject) {
            snapVideo.srcObject.getTracks().forEach(track => track.stop());
        }
    }

    btnAddStudentModal.addEventListener('click', openModal);
    btnQuickRegister.addEventListener('click', openModal);
    btnCloseModal.addEventListener('click', closeModal);
    btnCancelModal.addEventListener('click', closeModal);

    // Snap Camera view inside modal
    async function initSnapCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            snapVideo.srcObject = stream;
        } catch (err) {
            console.log("Snap camera unavailable");
        }
    }

    snapModeCamera.addEventListener('click', () => {
        snapModeCamera.classList.add('active');
        snapModeFile.classList.remove('active');
        snapBoxCamera.classList.remove('hidden');
        snapBoxFile.classList.add('hidden');
    });

    snapModeFile.addEventListener('click', () => {
        snapModeFile.classList.add('active');
        snapModeCamera.classList.remove('active');
        snapBoxFile.classList.remove('hidden');
        snapBoxCamera.classList.add('hidden');
    });

    let capturedBase64Image = null;

    btnCaptureSnapshot.addEventListener('click', () => {
        if (!snapVideo.videoWidth) return;
        snapCanvas.width = snapVideo.videoWidth;
        snapCanvas.height = snapVideo.videoHeight;
        const ctx = snapCanvas.getContext('2d');
        ctx.drawImage(snapVideo, 0, 0, snapCanvas.width, snapCanvas.height);
        capturedBase64Image = snapCanvas.toDataURL('image/jpeg', 0.9);
        regPreviewImg.src = capturedBase64Image;
        regPreviewBox.classList.remove('hidden');
        showToast('Face snapshot captured!', 'success');
    });

    regFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                capturedBase64Image = evt.target.result;
                regPreviewImg.src = capturedBase64Image;
                regPreviewBox.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const name = document.getElementById('reg-name').value.trim();
        const roll_no = document.getElementById('reg-roll').value.trim();
        const department = document.getElementById('reg-dept').value;
        const email = document.getElementById('reg-email').value.trim();

        if (!capturedBase64Image) {
            showToast('Please capture a face photo or upload an image file.', 'warning');
            return;
        }

        const submitBtn = document.getElementById('btn-submit-student');
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Saving & Training...`;

        try {
            const response = await fetch('/api/students', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name, roll_no, department, email, image: capturedBase64Image
                })
            });

            const resData = await response.json();
            if (resData.status === 'success') {
                showToast(resData.message, 'success');
                closeModal();
                registerForm.reset();
                regPreviewBox.classList.add('hidden');
                capturedBase64Image = null;
                loadStudentGrid();
                loadStats();
            } else {
                showToast(resData.message || 'Failed to register student.', 'error');
            }
        } catch (err) {
            showToast('Error connecting to server.', 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `Save & Train Student`;
        }
    });

    // --- Trigger Model Retrain ---
    async function triggerModelTrain() {
        showToast('Retraining AI Face Model...', 'warning');
        try {
            const res = await fetch('/api/train', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'success') {
                showToast(data.message, 'success');
                loadStats();
                if (state.activeTab === 'tab-analytics') loadAnalyticsTab();
            } else {
                showToast(data.message, 'error');
            }
        } catch (err) {
            showToast('Failed to trigger model retraining', 'error');
        }
    }

    btnTriggerTrain.addEventListener('click', triggerModelTrain);
    btnFullRetrain.addEventListener('click', triggerModelTrain);

    // Initial App Load
    initWebcam();
    loadStats();
});
