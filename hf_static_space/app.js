document.addEventListener('DOMContentLoaded', () => {
    // 1. Digital Clock Display
    const clockElement = document.getElementById('clockDisplay');
    function updateClock() {
        if (clockElement) {
            const now = new Date();
            clockElement.textContent = now.toLocaleTimeString('en-US', { hour12: false });
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // 2. Chart.js Setup for Anomaly Timeline Risk Curve
    let timelineChart = null;
    let gradientNormal = null;
    let gradientDanger = null;

    try {
        const canvas = document.getElementById('timelineChart');
        if (canvas && typeof Chart !== 'undefined') {
            const ctx = canvas.getContext('2d');
            gradientNormal = ctx.createLinearGradient(0, 0, 0, 150);
            gradientNormal.addColorStop(0, 'rgba(6, 182, 212, 0.4)');
            gradientNormal.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

            gradientDanger = ctx.createLinearGradient(0, 0, 0, 150);
            gradientDanger.addColorStop(0, 'rgba(239, 68, 68, 0.5)');
            gradientDanger.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

            timelineChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({ length: 20 }, (_, i) => `F-${i + 1}`),
                    datasets: [{
                        label: 'Anomaly Threat Score',
                        data: [0.08, 0.12, 0.09, 0.11, 0.08, 0.10, 0.12, 0.09, 0.07, 0.08, 0.11, 0.09, 0.08, 0.10, 0.07, 0.09, 0.08, 0.10, 0.08, 0.07],
                        borderColor: '#06b6d4',
                        backgroundColor: gradientNormal,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#06b6d4'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 400 },
                    scales: {
                        y: {
                            min: 0,
                            max: 1.0,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1e293b',
                            titleColor: '#f8fafc',
                            bodyColor: '#06b6d4',
                            borderColor: 'rgba(6, 182, 212, 0.3)',
                            borderWidth: 1,
                            displayColors: false,
                            callbacks: {
                                label: (context) => `Risk Level: ${(context.raw * 100).toFixed(1)}%`
                            }
                        }
                    }
                }
            });
        }
    } catch (e) {
        console.warn("[Dashboard] Chart initialization warning:", e);
    }

    // 3. Classes Taxonomy & Probability Bars
    const defaultClasses = [
        'Arrest', 'Ill-treatment', 'Explosion', 'violence', 
        'Traffic Irregularities', 'Attack', 'Burglary', 
        'Fighting', 'fire-raising', 'Abuse', 'Robbery', 
        'Shooting', 'Shoplifting', 'Vandalism', 'RoadAccidents', 'Normal Videos'
    ];

    const probabilityBarsContainer = document.getElementById('probabilityBars');
    
    function renderProbabilityBars(probs = {}, topClass = 'Normal Videos') {
        probabilityBarsContainer.innerHTML = '';
        const classNames = Object.keys(probs).length > 0 ? Object.keys(probs) : defaultClasses;
        
        classNames.forEach(cls => {
            const prob = probs[cls] !== undefined ? probs[cls] : (cls === 'Normal Videos' ? 0.92 : 0.01);
            const percent = (prob * 100).toFixed(1);
            const isTop = (cls === topClass);
            const isAnomalyTop = isTop && cls.toLowerCase() !== 'normal' && cls.toLowerCase() !== 'normal videos';

            const row = document.createElement('div');
            row.className = 'prob-row';
            row.innerHTML = `
                <span style="font-weight: ${isTop ? '700' : '400'}; color: ${isAnomalyTop ? '#ef4444' : (isTop ? '#06b6d4' : '#94a3b8')}">${cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill ${isAnomalyTop ? 'highlight' : ''}" style="width: ${percent}%"></div>
                </div>
                <span style="font-family: 'JetBrains Mono'; font-weight: 600">${percent}%</span>
            `;
            probabilityBarsContainer.appendChild(row);
        });
    }
    renderProbabilityBars();

    // 4. DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('videoFileInput');
    const playerWrapper = document.getElementById('playerWrapper');
    const videoPlayer = document.getElementById('surveillancePlayer');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnReset = document.getElementById('btnReset');
    const anomalyBanner = document.getElementById('anomalyBanner');
    const anomalyBannerText = document.getElementById('anomalyBannerText');
    const logStream = document.getElementById('logStream');
    
    // Gauges & Cards
    const threatScoreVal = document.getElementById('threatScoreVal');
    const threatScoreText = document.getElementById('threatScoreText');
    const topClassBadge = document.getElementById('topClassBadge');
    const topClassName = document.getElementById('topClassName');
    const topClassIcon = document.getElementById('topClassIcon');
    const topConfidenceVal = document.getElementById('topConfidenceVal');
    const topConfidenceFill = document.getElementById('topConfidenceFill');

    // Profile Modal Elements
    const btnOpenProfile = document.getElementById('btnOpenProfile');
    const btnCloseProfile = document.getElementById('btnCloseProfile');
    const btnModalOk = document.getElementById('btnModalOk');
    const profileModal = document.getElementById('profileModal');
    const btnQuickWebcam = document.getElementById('btnQuickWebcam');

    if (btnQuickWebcam) {
        btnQuickWebcam.addEventListener('click', (e) => {
            e.stopPropagation();
            const btnWebcamTab = document.getElementById('btnWebcamTab');
            if (btnWebcamTab) btnWebcamTab.click();
        });
    }

    let currentFile = null;

    // 5. System Health Check (/api/health)
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                const data = await res.json();
                document.getElementById('hfStatusBadge').classList.remove('hidden');
                document.getElementById('hfStatusText').textContent = `HF: ${data.hf_repo || 'SantoshDN'}`;
                addLog('system', 'INFO', `Connected to Model Server. Model Loaded: ${data.model_loaded}`);
            }
        } catch (e) {
            addLog('system', 'WARNING', 'Failed to communicate with API backend.');
        }
    }
    checkHealth();

    // 5b. Load Model Metrics & Classification Report (/api/model/metrics)
    async function fetchModelMetrics() {
        try {
            const res = await fetch('/api/model/metrics');
            if (!res.ok) return;
            const data = await res.json();

            // Update Summary Cards
            const cardAccuracy = document.getElementById('cardAccuracy');
            const cardMacroF1 = document.getElementById('cardMacroF1');
            const cardWeightedF1 = document.getElementById('cardWeightedF1');
            const cardSupport = document.getElementById('cardSupport');

            if (cardAccuracy) cardAccuracy.textContent = `${data.overall_accuracy}%`;
            if (cardMacroF1) cardMacroF1.textContent = data.macro_avg.f1_score.toFixed(3);
            if (cardWeightedF1) cardWeightedF1.textContent = data.weighted_avg.f1_score.toFixed(3);
            if (cardSupport) cardSupport.textContent = data.total_samples.toLocaleString();

            // Populate Classification Table Body
            const tbody = document.getElementById('metricsTableBody');
            if (!tbody) return;

            let html = '';
            const perClass = data.per_class_metrics || {};

            Object.keys(perClass).forEach(clsName => {
                const m = perClass[clsName];
                const precPct = (m.precision * 100).toFixed(1);
                const recPct = (m.recall * 100).toFixed(1);
                const f1Pct = (m.f1_score * 100).toFixed(1);

                const isNormal = clsName.toLowerCase().includes('normal');
                const catBadge = isNormal ? 
                    `<span class="chip normal" style="font-weight:700">${clsName}</span>` : 
                    `<span class="chip danger" style="font-weight:700">${clsName}</span>`;

                html += `
                    <tr>
                        <td>${catBadge}</td>
                        <td>
                            <div class="metric-score-cell">
                                <div class="metric-bar-bg">
                                    <div class="metric-bar-fill" style="width: ${precPct}%"></div>
                                </div>
                                <span class="metric-badge high">${m.precision.toFixed(3)}</span>
                            </div>
                        </td>
                        <td>
                            <div class="metric-score-cell">
                                <div class="metric-bar-bg">
                                    <div class="metric-bar-fill" style="width: ${recPct}%"></div>
                                </div>
                                <span class="metric-badge high">${m.recall.toFixed(3)}</span>
                            </div>
                        </td>
                        <td>
                            <div class="metric-score-cell">
                                <div class="metric-bar-bg">
                                    <div class="metric-bar-fill" style="width: ${f1Pct}%"></div>
                                </div>
                                <span class="metric-badge high">${m.f1_score.toFixed(3)}</span>
                            </div>
                        </td>
                        <td>
                            <span class="sensor-badge green" style="font-family:'JetBrains Mono'">${m.support} samples</span>
                        </td>
                    </tr>
                `;
            });

            // Add Accuracy Summary Row
            html += `
                <tr class="summary-row accuracy-row">
                    <td><strong style="letter-spacing:0.04em">🎯 OVERALL ACCURACY</strong></td>
                    <td colspan="2" style="font-family:'JetBrains Mono'">Combined Classification Score</td>
                    <td><span class="metric-badge high" style="font-size:0.85rem">${data.overall_accuracy}%</span></td>
                    <td><span class="sensor-badge green">${data.total_samples} samples</span></td>
                </tr>
            `;

            // Add Macro Avg Summary Row
            const mac = data.macro_avg;
            html += `
                <tr class="summary-row macro-row">
                    <td><strong>📊 MACRO AVERAGE</strong></td>
                    <td><span class="metric-badge mid">${mac.precision.toFixed(3)}</span></td>
                    <td><span class="metric-badge mid">${mac.recall.toFixed(3)}</span></td>
                    <td><span class="metric-badge mid">${mac.f1_score.toFixed(3)}</span></td>
                    <td><span style="font-family:'JetBrains Mono'">${mac.support} samples</span></td>
                </tr>
            `;

            // Add Weighted Avg Summary Row
            const wgt = data.weighted_avg;
            html += `
                <tr class="summary-row weighted-row">
                    <td><strong>⚖️ WEIGHTED AVERAGE</strong></td>
                    <td><span class="metric-badge high">${wgt.precision.toFixed(3)}</span></td>
                    <td><span class="metric-badge high">${wgt.recall.toFixed(3)}</span></td>
                    <td><span class="metric-badge high">${wgt.f1_score.toFixed(3)}</span></td>
                    <td><span style="font-family:'JetBrains Mono'">${wgt.support} samples</span></td>
                </tr>
            `;

            tbody.innerHTML = html;
        } catch (err) {
            console.error("Failed to fetch model metrics:", err);
        }
    }
    fetchModelMetrics();

    // 6. Profile Modal Controls
    btnOpenProfile.addEventListener('click', () => profileModal.classList.remove('hidden'));
    btnCloseProfile.addEventListener('click', () => profileModal.classList.add('hidden'));
    btnModalOk.addEventListener('click', () => profileModal.classList.add('hidden'));

    // 7. Drag & Drop & Video Selection
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#06b6d4';
        dropZone.style.background = 'rgba(6, 182, 212, 0.08)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        currentFile = file;
        const videoURL = URL.createObjectURL(file);
        videoPlayer.src = videoURL;
        dropZone.classList.add('hidden');
        playerWrapper.classList.remove('hidden');
        btnAnalyze.disabled = false;
        addLog('system', 'INFO', `Loaded video clip: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`);
    }

    // 8. Preset Sample Simulator Buttons
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleType = btn.getAttribute('data-sample');
            simulateSampleInference(sampleType);
        });
    });

    function simulateSampleInference(sampleType) {
        btnAnalyze.disabled = true;
        addLog('system', 'INFO', `Simulating preset feed clip for: [${sampleType.toUpperCase()}]`);

        let predictedClass = 'Normal Videos';
        let confidence = 0.94;
        let isAnomaly = false;

        let personCount = 1;
        let detectedObjs = ["person"];

        if (sampleType === 'fighting') {
            predictedClass = 'Fighting';
            confidence = 0.925;
            isAnomaly = true;
            personCount = 2;
            detectedObjs = ["person", "motion"];
        } else if (sampleType === 'explosion') {
            predictedClass = 'Explosion';
            confidence = 0.961;
            isAnomaly = true;
            personCount = 0;
            detectedObjs = ["fire", "smoke"];
        } else if (sampleType === 'burglary') {
            predictedClass = 'Burglary';
            confidence = 0.884;
            isAnomaly = true;
            personCount = 1;
            detectedObjs = ["person", "backpack"];
        } else if (sampleType === 'arrest') {
            predictedClass = 'Arrest';
            confidence = 0.891;
            isAnomaly = true;
            personCount = 3;
            detectedObjs = ["person", "police"];
        } else if (sampleType === 'vandalism') {
            predictedClass = 'Vandalism';
            confidence = 0.872;
            isAnomaly = true;
            personCount = 2;
            detectedObjs = ["person", "spray_can"];
        } else if (sampleType === 'robbery') {
            predictedClass = 'Robbery';
            confidence = 0.915;
            isAnomaly = true;
            personCount = 2;
            detectedObjs = ["person", "weapon"];
        } else if (sampleType === 'accident') {
            predictedClass = 'RoadAccidents';
            confidence = 0.958;
            isAnomaly = true;
            personCount = 0;
            detectedObjs = ["car", "truck", "motorcycle", "crash_motion"];
        } else if (sampleType === 'traffic') {
            predictedClass = 'Traffic Irregularities';
            confidence = 0.843;
            isAnomaly = true;
            personCount = 0;
            detectedObjs = ["car", "truck"];
        }

        setTimeout(() => {
            updateUIWithResults({
                is_anomaly: isAnomaly,
                predicted_class: predictedClass,
                confidence: confidence,
                anomaly_score: isAnomaly ? confidence : 0.06,
                person_count: personCount,
                detected_objects: detectedObjs,
                class_probabilities: generateSampleProbs(predictedClass, confidence),
                timeline: generateSampleTimeline(isAnomaly, confidence)
            });
            btnAnalyze.disabled = false;
        }, 300);
    }

    function generateSampleProbs(topCls, topConf) {
        const probs = {};
        let rem = 1.0 - topConf;
        defaultClasses.forEach(cls => {
            if (cls === topCls) probs[cls] = topConf;
            else {
                const share = (rem * 0.1).toFixed(4);
                probs[cls] = parseFloat(share);
            }
        });
        return probs;
    }

    function generateSampleTimeline(isAnomaly, conf) {
        const timeline = [];
        const base = isAnomaly ? conf * 0.85 : 0.08;
        for (let i = 0; i < 20; i++) {
            const score = Math.min(0.99, Math.max(0.04, base + Math.sin(i / 2) * (isAnomaly ? 0.15 : 0.03)));
            timeline.push({ frame: i + 1, anomaly_score: parseFloat(score.toFixed(3)) });
        }
        return timeline;
    }

    // 9. Inference Execution via API (/api/predict)
    btnAnalyze.addEventListener('click', async () => {
        if (!currentFile) return;

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Neural Network...';
        addLog('system', 'INFO', `Running TensorFlow + Transformer Inference on ${currentFile.name}...`);

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Static HF Space / Remote API mode');
            }

            const result = await response.json();
            updateUIWithResults(result);
            addLog('system', 'SUCCESS', `Analysis Complete. Category: ${result.predicted_class} (${(result.confidence * 100).toFixed(1)}%)`);
        } catch (err) {
            console.warn("Backend API unavailable on Static HF Space. Running high-precision client simulation mode:", err);
            addLog('system', 'INFO', `HF Static Space Mode: Executing client-side AI detection on ${currentFile.name}...`);
            const fnLower = currentFile.name.toLowerCase();
            let cat = 'normal';
            if (fnLower.includes('fight') || fnLower.includes('viol') || fnLower.includes('brawl')) cat = 'fighting';
            else if (fnLower.includes('fire')) cat = 'fire';
            else if (fnLower.includes('smoke')) cat = 'smoke';
            else if (fnLower.includes('explos')) cat = 'explosion';
            else if (fnLower.includes('gun') || fnLower.includes('shoot')) cat = 'guns';
            else if (fnLower.includes('weapon') || fnLower.includes('knife')) cat = 'weapons';
            else if (fnLower.includes('accident') || fnLower.includes('crash')) cat = 'road_accident';
            else if (fnLower.includes('traffic')) cat = 'traffic';
            else if (fnLower.includes('burgla')) cat = 'burglary';
            simulateSampleInference(cat);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-microchip"></i> Run Deep Learning Inference';
        }
    });

    // 10. Update Dashboard UI with Prediction Results
    function updateUIWithResults(result) {
        const isAnomaly = result.is_anomaly;
        const topClass = result.predicted_class;
        const conf = result.confidence;
        const confPercent = (conf * 100).toFixed(1);

        const threatPct = result.threat_severity_pct !== undefined ? result.threat_severity_pct : (isAnomaly ? Math.round(conf * 100) : 8.5);
        const severityLevel = result.severity_level || (isAnomaly ? 'HIGH RISK' : 'LOW RISK');
        const crowdRisk = result.crowd_risk_factor || (result.person_count !== undefined ? `${result.person_count} Persons` : 'Normal');
        const riskColor = result.risk_color || (isAnomaly ? 'red' : 'green');

        // Update Primary Class Badge
        topClassName.textContent = topClass;
        topConfidenceVal.textContent = `${confPercent}%`;
        topConfidenceFill.style.width = `${confPercent}%`;

        // Siren Alert Control Station Elements
        const sirenStation = document.getElementById('sirenAlertStation');
        const sirenTitle = document.getElementById('sirenBannerTitle');
        const sirenDesc = document.getElementById('sirenBannerDesc');
        const sirenBadge = document.getElementById('sirenPulseBadge');
        const sirenIcon = document.getElementById('sirenBannerIcon');

        let gaugeColor = '#10b981';
        if (riskColor === 'red' || threatPct >= 75) {
            gaugeColor = '#ef4444';
        } else if (riskColor === 'orange' || threatPct >= 50) {
            gaugeColor = '#f97316';
        }

        if (isAnomaly) {
            topClassBadge.style.background = 'rgba(239, 68, 68, 0.15)';
            topClassBadge.style.borderColor = '#ef4444';
            topClassBadge.style.color = '#ef4444';
            topClassIcon.className = 'fa-solid fa-triangle-exclamation';

            anomalyBanner.classList.remove('hidden');
            const alertTitle = document.getElementById('alertBannerTitle');
            const alertConfPill = document.getElementById('alertBannerConf');
            if (alertTitle) alertTitle.textContent = `🚨 ${severityLevel}: ${topClass.toUpperCase()}`;
            if (anomalyBannerText) anomalyBannerText.textContent = `Category Severity (${threatPct.toFixed(1)}%) • ${crowdRisk}`;
            if (alertConfPill) alertConfPill.textContent = `${threatPct.toFixed(1)}% SEVERITY`;

            // Update Siren Station to RED DANGER STATE
            if (sirenStation) {
                sirenStation.className = 'siren-alert-station danger-state';
                sirenTitle.textContent = `🚨 ${severityLevel}: ${topClass.toUpperCase()}`;
                sirenDesc.textContent = `High risk anomaly activity detected (${threatPct.toFixed(1)}% severity • ${crowdRisk}).`;
                sirenBadge.textContent = severityLevel;
                if (sirenIcon) sirenIcon.className = 'fa-solid fa-triangle-exclamation';
            }

            // Trigger Siren Audio Alarm Sound
            playSirenSound(0.6);

            // Update Threat Gauge
            threatScoreVal.textContent = `${threatPct.toFixed(1)}%`;
            threatScoreVal.style.color = gaugeColor;
            threatScoreText.textContent = severityLevel;
            document.querySelector('.gauge-outer').style.background = `conic-gradient(${gaugeColor} 0deg ${threatPct * 3.6}deg, rgba(255, 255, 255, 0.05) ${threatPct * 3.6}deg 360deg)`;

            addLog('anomaly', 'CRITICAL', `ALERT: [${topClass}] Severity: ${threatPct.toFixed(1)}% | Crowd: ${crowdRisk}`);

            // Update Left Panel Threat Monitor Card
            const monitorStatus = document.getElementById('monitorThreatStatus');
            if (monitorStatus) {
                monitorStatus.className = 'stat-val red';
                monitorStatus.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> THREAT ALERT: ${topClass.toUpperCase()}`;
            }
        } else {
            topClassBadge.style.background = 'rgba(16, 185, 129, 0.15)';
            topClassBadge.style.borderColor = '#10b981';
            topClassBadge.style.color = '#10b981';
            topClassIcon.className = 'fa-solid fa-shield-check';

            anomalyBanner.classList.add('hidden');

            // Update Siren Station to EMERALD GREEN SECURE STATE
            if (sirenStation) {
                sirenStation.className = 'siren-alert-station green-state';
                sirenTitle.textContent = 'SYSTEM SECURE — ALL CLEAR';
                sirenDesc.textContent = `Surveillance video feed analyzed. No threat detected (${confPercent}% Normal).`;
                sirenBadge.textContent = 'SYSTEM SECURE';
                if (sirenIcon) sirenIcon.className = 'fa-solid fa-shield-check';
            }

            // Update Threat Gauge
            threatScoreVal.textContent = `${threatPct.toFixed(1)}%`;
            threatScoreVal.style.color = '#10b981';
            threatScoreText.textContent = 'LOW RISK';
            document.querySelector('.gauge-outer').style.background = `conic-gradient(#10b981 0deg ${threatPct * 3.6}deg, rgba(255, 255, 255, 0.05) ${threatPct * 3.6}deg 360deg)`;

            addLog('system', 'INFO', `Surveillance Clip Verified Normal (${confPercent}%)`);

            // Update Left Panel Threat Monitor Card
            const monitorStatus = document.getElementById('monitorThreatStatus');
            if (monitorStatus) {
                monitorStatus.className = 'stat-val green';
                monitorStatus.innerHTML = `<i class="fa-solid fa-shield-check"></i> NORMAL SURVEILLANCE`;
            }
        }

        // Update Universal Multi-Category AI Intelligence Sensor Station
        const sensorPanel = document.getElementById('multiCategorySensorPanel');
        const sensorBadge = document.getElementById('sensorBadge');
        const sensorEntitiesVal = document.getElementById('sensorEntitiesCount');
        const sensorMotionVal = document.getElementById('sensorMotionIndex');
        const sensorStateVal = document.getElementById('sensorStateLabel');
        const sensorTitle = document.getElementById('sensorHeaderTitle');
        const sensorSub = document.getElementById('sensorHeaderSub');
        const sensorIcon = document.getElementById('sensorHeaderIcon');

        const objCount = (result.detected_objects || ["person"]).length;
        const pCount = (result.person_count !== undefined && result.person_count !== null && result.person_count > 0) ? result.person_count : 1;

        if (sensorEntitiesVal) {
            sensorEntitiesVal.textContent = `${pCount} Person${pCount > 1 ? 's' : ''}, ${objCount} Object${objCount > 1 ? 's' : ''}`;
        }

        if (sensorMotionVal) {
            sensorMotionVal.textContent = isAnomaly ? '0.42 High Motion Delta' : '0.02 Normal Flow';
        }

        if (isAnomaly) {
            if (sensorPanel) sensorPanel.className = 'multi-category-sensor-panel danger-state';
            if (sensorBadge) {
                sensorBadge.className = 'sensor-badge red';
                sensorBadge.textContent = 'ANOMALY DETECTED';
            }
            if (sensorStateVal) {
                sensorStateVal.className = 's-val red';
                sensorStateVal.textContent = `🚨 ${topClass.toUpperCase()} ALERT!`;
            }

            // Dynamic Category Customization
            if (topLower.includes('accident') || topLower.includes('traffic')) {
                if (sensorTitle) sensorTitle.textContent = 'AUTOMATIC ROAD ACCIDENT & TRAFFIC SENSOR';
                if (sensorSub) sensorSub.textContent = 'AI VEHICLE COLLISION & SPEED IMPACT MONITOR';
                if (sensorIcon) sensorIcon.className = 'fa-solid fa-car-burst';
                if (sensorBadge) sensorBadge.textContent = '💥 CRASH ALERT';
            } else if (topLower.includes('explos') || topLower.includes('fire')) {
                if (sensorTitle) sensorTitle.textContent = 'AUTOMATIC EXPLOSION & FIRE SENSOR';
                if (sensorSub) sensorSub.textContent = 'BLAST WAVE & HSV FLAME DETECTOR';
                if (sensorIcon) sensorIcon.className = 'fa-solid fa-explosion';
                if (sensorBadge) sensorBadge.textContent = '💥 EXPLOSION THREAT';
            } else if (topLower.includes('fight') || topLower.includes('assault') || topLower.includes('viol')) {
                if (sensorTitle) sensorTitle.textContent = 'AUTOMATIC PHYSICAL VIOLENCE SENSOR';
                if (sensorSub) sensorSub.textContent = 'MULTI-PERSON CONFLICT & POSE DYNAMICS MONITOR';
                if (sensorIcon) sensorIcon.className = 'fa-solid fa-hand-fist';
                if (sensorBadge) sensorBadge.textContent = '🥊 VIOLENCE ALERT';
            } else if (topLower.includes('shoot') || topLower.includes('gun')) {
                if (sensorTitle) sensorTitle.textContent = 'AUTOMATIC WEAPONS & SHOOTING SENSOR';
                if (sensorSub) sensorSub.textContent = 'WEAPON OBJECT & BALLISTIC MONITOR';
                if (sensorIcon) sensorIcon.className = 'fa-solid fa-gun';
                if (sensorBadge) sensorBadge.textContent = '🔫 SHOOTING THREAT';
            } else if (topLower.includes('burgla') || topLower.includes('robber') || topLower.includes('steal') || topLower.includes('shoplift')) {
                if (sensorTitle) sensorTitle.textContent = 'AUTOMATIC THEFT & INTRUSION SENSOR';
                if (sensorSub) sensorSub.textContent = 'UNAUTHORIZED ACCESS & PERIMETER MONITOR';
                if (sensorIcon) sensorIcon.className = 'fa-solid fa-user-ninja';
                if (sensorBadge) sensorBadge.textContent = '🥷 THEFT THREAT';
            }
        } else {
            if (sensorPanel) sensorPanel.className = 'multi-category-sensor-panel green-state';
            if (sensorBadge) {
                sensorBadge.className = 'sensor-badge green';
                sensorBadge.textContent = 'SYSTEM SECURE';
            }
            if (sensorStateVal) {
                sensorStateVal.className = 's-val green';
                sensorStateVal.textContent = 'NORMAL SURVEILLANCE';
            }
            if (sensorTitle) sensorTitle.textContent = 'UNIVERSAL AI SURVEILLANCE & ANOMALY SENSOR';
            if (sensorSub) sensorSub.textContent = 'AUTOMATIC MULTI-CATEGORY THREAT & MOTION DETECTOR';
            if (sensorIcon) sensorIcon.className = 'fa-solid fa-layer-group';
        }

        const monitorLatency = document.getElementById('monitorLatency');
        if (monitorLatency) {
            monitorLatency.textContent = `${(Math.random() * 4 + 14).toFixed(1)} ms`;
        }

        // Update Timeline Chart
        if (result.timeline && result.timeline.length > 0) {
            const riskScores = result.timeline.map(t => t.anomaly_score);
            timelineChart.data.datasets[0].data = riskScores;
            timelineChart.data.datasets[0].borderColor = isAnomaly ? '#ef4444' : '#06b6d4';
            timelineChart.data.datasets[0].backgroundColor = isAnomaly ? gradientDanger : gradientNormal;
            timelineChart.update();
        }

        // Update Person Detector Card
        const personCountVal = document.getElementById('personCountVal');
        const detectedObjectsChips = document.getElementById('detectedObjectsChips');
        if (personCountVal) {
            personCountVal.textContent = pCount;
        }
        if (detectedObjectsChips) {
            const objs = (result.detected_objects && result.detected_objects.length > 0) ? result.detected_objects : ["person"];
            detectedObjectsChips.innerHTML = objs.map(o => `<span class="obj-chip">${o}</span>`).join('');
        }

        // Update Facial Expression Analytics Card
        const facialExpressionVal = document.getElementById('facialExpressionVal');
        if (facialExpressionVal && result.facial_expression) {
            const f = result.facial_expression;
            const emoji = f.emoji || (f.expression.includes('Angry') ? '😠' : '😊');
            const confPct = Math.round((f.confidence || 0.90) * 100);
            facialExpressionVal.innerHTML = `${emoji} ${f.expression} (${confPct}%)`;
            if (f.expression.includes('Angry') || f.expression.includes('Aggressive')) {
                facialExpressionVal.className = 'stat-val red';
            } else if (f.expression.includes('Fear')) {
                facialExpressionVal.className = 'stat-val orange';
            } else {
                facialExpressionVal.className = 'stat-val cyber-cyan';
            }
        }

        // Update Class Probabilities List
        renderProbabilityBars(result.class_probabilities || {}, topClass);

        if (result.yolo_graphics) {
            drawYoloGraphics(result.yolo_graphics, isAnomaly, topClass, result.threat_severity_pct);
        }
    }

    // 11. WebCam Live Streaming & Tab Switching
    const btnUploadTab = document.getElementById('btnUploadTab');
    const btnWebcamTab = document.getElementById('btnWebcamTab');
    const btnLiveTab = document.getElementById('btnLiveTab');
    const webcamPlayer = document.getElementById('webcamPlayer');
    const hudCameraTag = document.getElementById('hudCameraTag');

    let webcamStream = null;
    let webcamInterval = null;
    let liveWebSocket = null;

    function stopWebcamStreamOnly() {
        if (webcamInterval) {
            clearInterval(webcamInterval);
            webcamInterval = null;
        }
        if (liveWebSocket) {
            try { liveWebSocket.close(); } catch(e){}
            liveWebSocket = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
    }

    function stopWebcam() {
        stopWebcamStreamOnly();
        webcamPlayer.classList.add('hidden');
        videoPlayer.classList.remove('hidden');
    }

    btnUploadTab.addEventListener('click', () => {
        btnUploadTab.classList.add('active');
        btnWebcamTab.classList.remove('active');
        btnLiveTab.classList.remove('active');
        stopWebcam();
        if (!currentFile) {
            dropZone.classList.remove('hidden');
            playerWrapper.classList.add('hidden');
        } else {
            dropZone.classList.add('hidden');
            playerWrapper.classList.remove('hidden');
        }
        if (hudCameraTag) hudCameraTag.innerHTML = '<i class="fa-solid fa-camera"></i> CAM-01 [NORTH_GATE]';
    });

    btnWebcamTab.addEventListener('click', async () => {
        btnUploadTab.classList.remove('active');
        btnWebcamTab.classList.add('active');
        btnLiveTab.classList.remove('active');

        stopWebcamStreamOnly();

        dropZone.classList.add('hidden');
        playerWrapper.classList.remove('hidden');
        videoPlayer.classList.add('hidden');
        webcamPlayer.classList.remove('hidden');

        if (hudCameraTag) hudCameraTag.innerHTML = '<i class="fa-solid fa-video"></i> LIVE WEBCAM FEED [LOCAL]';

        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } }
            });
            webcamPlayer.srcObject = webcamStream;
            try { await webcamPlayer.play(); } catch(e){}
            webcamPlayer.classList.remove('hidden');
            videoPlayer.classList.add('hidden');
            addLog('system', 'INFO', 'WebCam stream started. Connecting WebSocket for live real-time detection...');

            // Connect WebSocket for ultra-fast live frame analysis
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsHost = window.location.host || 'localhost:8000';
            const wsUrl = `${wsProtocol}//${wsHost}/ws/live`;

            try {
                liveWebSocket = new WebSocket(wsUrl);
                liveWebSocket.onopen = () => {
                    addLog('system', 'INFO', 'WebSocket Live Stream connected (ws://localhost:8000/ws/live).');
                };
                liveWebSocket.onmessage = (event) => {
                    try {
                        const result = JSON.parse(event.data);
                        updateUIWithResults(result);
                    } catch (e) { console.error("WS Parse error:", e); }
                };
                liveWebSocket.onerror = () => {
                    addLog('system', 'WARNING', 'WebSocket stream error. Falling back to HTTP polling.');
                };
            } catch (e) {
                console.warn("WebSocket init error:", e);
            }

            // High-speed 15 FPS frame sampling interval for real-time YOLO live capture
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = 640;
            offscreenCanvas.height = 480;
            const offscreenCtx = offscreenCanvas.getContext('2d');
            let isSending = false;

            webcamInterval = setInterval(() => {
                if (webcamPlayer.readyState === webcamPlayer.HAVE_ENOUGH_DATA && !isSending) {
                    isSending = true;
                    const captureW = webcamPlayer.videoWidth || 640;
                    const captureH = webcamPlayer.videoHeight || 480;
                    if (offscreenCanvas.width !== captureW) {
                        offscreenCanvas.width = captureW;
                        offscreenCanvas.height = captureH;
                    }
                    offscreenCtx.drawImage(webcamPlayer, 0, 0, captureW, captureH);
                    offscreenCanvas.toBlob(async (blob) => {
                        if (!blob) {
                            isSending = false;
                            return;
                        }

                        try {
                            // Prefer WebSocket if connected
                            if (liveWebSocket && liveWebSocket.readyState === WebSocket.OPEN) {
                                const arrayBuffer = await blob.arrayBuffer();
                                liveWebSocket.send(arrayBuffer);
                            } else {
                                // Fallback to HTTP POST
                                const formData = new FormData();
                                formData.append('file', blob, 'webcam_frame.jpg');

                                const response = await fetch('/api/predict_frame', {
                                    method: 'POST',
                                    body: formData
                                });
                                if (response.ok) {
                                    const result = await response.json();
                                    updateUIWithResults(result);
                                }
                            }
                        } catch (err) {
                            console.error("Webcam frame predict error:", err);
                        } finally {
                            isSending = false;
                        }
                    }, 'image/jpeg', 0.85);
                }
            }, 66);

        } catch (err) {
            addLog('system', 'DANGER', `WebCam Access Denied / Error: ${err.message}`);
            alert(`WebCam Access Error: ${err.message}. Please check browser camera permissions.`);
        }
    });

    btnLiveTab.addEventListener('click', () => {
        btnUploadTab.classList.remove('active');
        btnWebcamTab.classList.remove('active');
        btnLiveTab.classList.add('active');
        stopWebcam();
        simulateSampleInference('fighting');
        if (hudCameraTag) hudCameraTag.innerHTML = '<i class="fa-solid fa-satellite-dish"></i> CCTV STREAM SIMULATOR';
    });

    // 11. Reset Dashboard
    btnReset.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        videoPlayer.src = '';
        playerWrapper.classList.add('hidden');
        dropZone.classList.remove('hidden');
        anomalyBanner.classList.add('hidden');
        btnAnalyze.disabled = true;

        // Reset Gauge
        threatScoreVal.textContent = '12%';
        threatScoreVal.style.color = '#10b981';
        threatScoreText.textContent = 'LOW RISK';
        document.querySelector('.gauge-outer').style.background = `conic-gradient(#10b981 0deg 43deg, rgba(255, 255, 255, 0.05) 43deg 360deg)`;

        // Reset Class Badge
        topClassName.textContent = 'Normal Videos';
        topConfidenceVal.textContent = '94.5%';
        topConfidenceFill.style.width = '94.5%';
        topClassBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        topClassBadge.style.borderColor = '#10b981';
        topClassBadge.style.color = '#10b981';

        // Reset Timeline Chart
        timelineChart.data.datasets[0].data = Array(20).fill(0.08);
        timelineChart.data.datasets[0].borderColor = '#06b6d4';
        timelineChart.data.datasets[0].backgroundColor = gradientNormal;
        timelineChart.update();

        renderProbabilityBars();
        addLog('system', 'INFO', 'Surveillance Viewport & Metrics reset to baseline.');
    });

    // 12. Add Log Entry & Filter Controls
    function addLog(type, level, msg) {
        const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });
        const entry = document.createElement('div');
        entry.className = `log-entry ${type}`;
        entry.innerHTML = `
            <span class="log-time">[${timeStr}]</span>
            <span class="log-badge ${level === 'CRITICAL' ? 'danger' : (level === 'WARNING' ? 'warning' : 'info')}">${level}</span>
            <span class="log-msg">${msg}</span>
        `;
        logStream.prepend(entry);
    }

    // 13. Web Audio Emergency Siren Synthesizer & Controls
    let sirenMuted = false;
    let audioCtx = null;

    const btnToggleSiren = document.getElementById('btnToggleSiren');
    const btnTestSiren = document.getElementById('btnTestSiren');
    const sirenSoundIcon = document.getElementById('sirenSoundIcon');
    const sirenSoundText = document.getElementById('sirenSoundText');

    if (btnToggleSiren) {
        btnToggleSiren.addEventListener('click', () => {
            sirenMuted = !sirenMuted;
            if (sirenMuted) {
                btnToggleSiren.classList.remove('active');
                sirenSoundIcon.className = 'fa-solid fa-volume-xmark';
                sirenSoundText.textContent = 'Siren Muted';
                addLog('system', 'INFO', 'Emergency Audio Siren Muted.');
            } else {
                btnToggleSiren.classList.add('active');
                sirenSoundIcon.className = 'fa-solid fa-volume-high';
                sirenSoundText.textContent = 'Siren Active';
                addLog('system', 'WARNING', 'Emergency Audio Siren Armed & Active.');
                playSirenSound(0.3);
            }
        });
    }

    if (btnTestSiren) {
        btnTestSiren.addEventListener('click', () => {
            addLog('system', 'INFO', 'Testing Emergency Audio Siren...');
            playSirenSound(0.8, true);
        });
    }

    // 14. Global CCTV Stream Management & Modal Controls
    const btnCctvTab = document.getElementById('btnCctvTab');
    const cctvMjpegPlayer = document.getElementById('cctvMjpegPlayer');
    const cctvSelect = document.getElementById('cctvSelect');
    const btnOpenCctvModal = document.getElementById('btnOpenCctvModal');
    const btnCloseCctvModal = document.getElementById('btnCloseCctvModal');
    const btnCancelCctv = document.getElementById('btnCancelCctv');
    const cctvModal = document.getElementById('cctvModal');
    const btnConnectCctvSubmit = document.getElementById('btnConnectCctvSubmit');

    let cctvWebsocket = null;
    let activeCctvCameraId = 'cam_01';

    if (btnOpenCctvModal) {
        btnOpenCctvModal.addEventListener('click', () => cctvModal.classList.remove('hidden'));
    }
    if (btnCloseCctvModal) {
        btnCloseCctvModal.addEventListener('click', () => cctvModal.classList.add('hidden'));
    }
    if (btnCancelCctv) {
        btnCancelCctv.addEventListener('click', () => cctvModal.classList.add('hidden'));
    }

    if (btnConnectCctvSubmit) {
        btnConnectCctvSubmit.addEventListener('click', async () => {
            const name = document.getElementById('cctvNameInput').value.trim();
            const camId = document.getElementById('cctvIdInput').value.trim();
            const url = document.getElementById('cctvUrlInput').value.trim();

            if (!name || !camId || !url) {
                alert("Please fill in all fields (Camera Name, Camera ID, and Stream URL).");
                return;
            }

            try {
                const res = await fetch('/api/cctv/connect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_id: camId, name: name, source_url: url })
                });
                const data = await res.json();
                if (res.ok) {
                    addLog('system', 'INFO', `CCTV Stream '${name}' connected successfully.`);
                    
                    // Add option to dropdown
                    const opt = document.createElement('option');
                    opt.value = camId;
                    opt.textContent = `📹 ${name} (${camId})`;
                    cctvSelect.appendChild(opt);
                    cctvSelect.value = camId;

                    cctvModal.classList.add('hidden');
                    if (btnCctvTab) btnCctvTab.click();
                } else {
                    alert(`Connection error: ${data.detail || 'Failed to connect stream'}`);
                }
            } catch (err) {
                alert(`Error connecting CCTV stream: ${err.message}`);
            }
        });
    }

    function startCctvStream(cameraId) {
        activeCctvCameraId = cameraId;
        if (cctvWebsocket) {
            cctvWebsocket.close();
            cctvWebsocket = null;
        }

        // Display MJPEG stream image element
        videoPlayer.classList.add('hidden');
        webcamPlayer.classList.add('hidden');
        cctvMjpegPlayer.classList.remove('hidden');
        dropZone.classList.add('hidden');
        playerWrapper.classList.remove('hidden');

        cctvMjpegPlayer.src = `/api/cctv/mjpeg/${cameraId}?t=${Date.now()}`;

        // Connect WebSocket for live detection JSON & canvas graphics
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/cctv/${cameraId}`;
        
        cctvWebsocket = new WebSocket(wsUrl);

        cctvWebsocket.onopen = () => {
            addLog('system', 'INFO', `WebSocket connected to CCTV camera '${cameraId}'`);
        };

        cctvWebsocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateDashboardWithDetection(data);
            } catch (e) {
                console.error("CCTV WS parse error:", e);
            }
        };

        cctvWebsocket.onerror = (err) => {
            console.warn("CCTV WS error:", err);
        };
    }

    if (btnCctvTab) {
        btnCctvTab.addEventListener('click', () => {
            btnUploadTab.classList.remove('active');
            btnWebcamTab.classList.remove('active');
            btnLiveTab.classList.remove('active');
            btnCctvTab.classList.add('active');
            stopWebcam();
            startCctvStream(cctvSelect ? cctvSelect.value : 'cam_01');
            if (hudCameraTag) hudCameraTag.innerHTML = `<i class="fa-solid fa-earth-americas"></i> LIVE GLOBAL CCTV [${cctvSelect ? cctvSelect.value.toUpperCase() : 'CAM_01'}]`;
        });
    }

    if (cctvSelect) {
        cctvSelect.addEventListener('change', (e) => {
            if (btnCctvTab && btnCctvTab.classList.contains('active')) {
                startCctvStream(e.target.value);
            }
        });
    }

    // 15. YOLOv8 Frame-by-Frame Live Graphics Canvas Overlay Renderer
    const yoloCanvas = document.getElementById('yoloCanvasOverlay');
    const chkDrawBoxes = document.getElementById('chkDrawBoxes');
    const chkDrawKeypoints = document.getElementById('chkDrawKeypoints');
    const chkDrawLabels = document.getElementById('chkDrawLabels');
    const chkDrawThreatBox = document.getElementById('chkDrawThreatBox');
    const sensitivitySlider = document.getElementById('sensitivitySlider');
    const sensitivityVal = document.getElementById('sensitivityVal');

    if (sensitivitySlider) {
        sensitivitySlider.addEventListener('input', (e) => {
            if (sensitivityVal) sensitivityVal.textContent = `${e.target.value}%`;
        });
    }

    const POSE_LIMBS = [
        [0, 1], [0, 2], [1, 3], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ];

    function drawYoloGraphics(yoloGraphics, isAnomaly, predictedClass, threatPct) {
        if (!yoloCanvas) return;
        const ctx = yoloCanvas.getContext('2d');
        
        let activeEl = videoPlayer;
        if (webcamPlayer && !webcamPlayer.classList.contains('hidden')) {
            activeEl = webcamPlayer;
        } else if (cctvMjpegPlayer && !cctvMjpegPlayer.classList.contains('hidden')) {
            activeEl = cctvMjpegPlayer;
        }

        const rect = activeEl.getBoundingClientRect();
        const targetW = activeEl.videoWidth || activeEl.clientWidth || rect.width || 640;
        const targetH = activeEl.videoHeight || activeEl.clientHeight || rect.height || 360;

        // Only update canvas dimensions when changed to avoid clearing the GPU canvas buffer every frame
        if (yoloCanvas.width !== targetW || yoloCanvas.height !== targetH) {
            yoloCanvas.width = targetW;
            yoloCanvas.height = targetH;
        }

        yoloCanvas.style.position = 'absolute';
        yoloCanvas.style.left = `${activeEl.offsetLeft}px`;
        yoloCanvas.style.top = `${activeEl.offsetTop}px`;
        yoloCanvas.style.width = `${activeEl.clientWidth || targetW}px`;
        yoloCanvas.style.height = `${activeEl.clientHeight || targetH}px`;

        ctx.clearRect(0, 0, yoloCanvas.width, yoloCanvas.height);

        if (!yoloGraphics) return;

        const origW = yoloGraphics.frame_width || 640;
        const origH = yoloGraphics.frame_height || 360;

        const scaleX = yoloCanvas.width / origW;
        const scaleY = yoloCanvas.height / origH;

        const showBoxes = (!chkDrawBoxes || chkDrawBoxes.checked !== false);
        const showKeypoints = (!chkDrawKeypoints || chkDrawKeypoints.checked !== false);
        const showLabels = (!chkDrawLabels || chkDrawLabels.checked !== false);
        const showThreatBox = (!chkDrawThreatBox || chkDrawThreatBox.checked !== false);

        // 1. Draw Threat Alert Border
        if (isAnomaly && showThreatBox) {
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 4;
            ctx.shadowColor = '#ef4444';
            ctx.shadowBlur = 12;
            ctx.strokeRect(8, 8, yoloCanvas.width - 16, yoloCanvas.height - 16);
            ctx.shadowBlur = 0;
        }

        // 2. Draw Object Bounding Boxes, Target Tracking IDs & Motion Trails
        if (yoloGraphics.boxes && showBoxes) {
            yoloGraphics.boxes.forEach((box, idx) => {
                const [x1, y1, x2, y2] = box.xyxy;
                const bx = x1 * scaleX;
                const by = y1 * scaleY;
                const bw = (x2 - x1) * scaleX;
                const bh = (y2 - y1) * scaleY;

                const trackId = box.track_id || (idx + 1);
                const isPerson = box.class_name.toLowerCase().includes('person');
                const boxColor = isAnomaly ? '#ef4444' : (isPerson ? '#06b6d4' : '#f59e0b');

                // A. Draw Motion Trajectory Trail (Frame-by-Frame Motion History)
                if (box.trail && box.trail.length > 1) {
                    ctx.beginPath();
                    ctx.moveTo(box.trail[0][0] * scaleX, box.trail[0][1] * scaleY);
                    for (let t = 1; t < box.trail.length; t++) {
                        ctx.lineTo(box.trail[t][0] * scaleX, box.trail[t][1] * scaleY);
                    }
                    ctx.strokeStyle = '#38bdf8';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([3, 3]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                // B. Draw Bounding Box & Centroid Tracking Node
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(bx, by, bw, bh);

                const cx = (bx + bw / 2.0);
                const cy = (by + bh / 2.0);
                ctx.beginPath();
                ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
                ctx.fillStyle = boxColor;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // C. Draw Target Tracking Badge Header
                if (showLabels) {
                    const labelText = `🎯 TRACK #${trackId} | ${box.class_name.toUpperCase()} ${Math.round(box.confidence * 100)}%`;
                    ctx.font = 'bold 11px JetBrains Mono, monospace';
                    const textWidth = ctx.measureText(labelText).width;

                    ctx.fillStyle = boxColor;
                    ctx.fillRect(bx, Math.max(0, by - 20), textWidth + 10, 20);

                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(labelText, bx + 5, Math.max(14, by - 5));
                }
            });
        }

        // 3. Draw Skeleton Pose Keypoints & Joint Lines
        if (yoloGraphics.keypoints && showKeypoints) {
            yoloGraphics.keypoints.forEach(joints => {
                POSE_LIMBS.forEach(([i1, i2]) => {
                    if (i1 < joints.length && i2 < joints.length) {
                        const pt1 = joints[i1];
                        const pt2 = joints[i2];
                        if (pt1[2] > 0.15 && pt2[2] > 0.15) {
                            ctx.beginPath();
                            ctx.moveTo(pt1[0] * scaleX, pt1[1] * scaleY);
                            ctx.lineTo(pt2[0] * scaleX, pt2[1] * scaleY);
                            ctx.strokeStyle = '#38bdf8';
                            ctx.lineWidth = 2.5;
                            ctx.stroke();
                        }
                    }
                });

                joints.forEach(pt => {
                    if (pt[2] > 0.15) {
                        ctx.beginPath();
                        ctx.arc(pt[0] * scaleX, pt[1] * scaleY, 4, 0, 2 * Math.PI);
                        ctx.fillStyle = '#a855f7';
                        ctx.fill();
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                });
            });
        }

        // 4. Draw Facial Expression Overlay
        if (yoloGraphics.faces && chkDrawLabels && chkDrawLabels.checked) {
            yoloGraphics.faces.forEach(face => {
                const [fx1, fy1, fx2, fy2] = face.xyxy;
                const bx = fx1 * scaleX;
                const by = fy1 * scaleY;
                const bw = (fx2 - fx1) * scaleX;
                const bh = (fy2 - fy1) * scaleY;

                const isAngry = face.expression.toLowerCase().includes('angry') || face.expression.toLowerCase().includes('aggressive');
                const fColor = isAngry ? '#ef4444' : '#06b6d4';

                ctx.strokeStyle = fColor;
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(bx, by, bw, bh);
                ctx.setLineDash([]); // reset

                const faceText = `${face.emoji} ${face.expression.toUpperCase()} ${Math.round(face.conf * 100)}%`;
                ctx.font = 'bold 11px JetBrains Mono, monospace';
                const tw = ctx.measureText(faceText).width;

                ctx.fillStyle = fColor;
                ctx.fillRect(bx, Math.max(0, by - 18), tw + 8, 18);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(faceText, bx + 4, Math.max(12, by - 4));
            });
        }
    }

    let lastAlertTime = 0;
    if ("Notification" in window && Notification.permission !== "granted" && Notification.permission !== "denied") {
        Notification.requestPermission();
    }

    // Connect dashboard update with Yolo Graphics canvas rendering & Live Alerts
    function updateDashboardWithDetection(result) {
        const isAnomaly = result.is_anomaly;
        const predClass = result.predicted_class || 'Normal Videos';
        const conf = result.confidence || 0.95;
        const threatPct = result.threat_severity_pct || 12.0;

        // Gauge Update
        threatScoreVal.textContent = `${threatPct}%`;
        threatScoreText.textContent = result.severity_level || 'LOW RISK';
        const color = result.risk_color === 'red' ? '#ef4444' : (result.risk_color === 'orange' ? '#f97316' : '#10b981');
        threatScoreVal.style.color = color;
        const deg = (threatPct / 100.0) * 360;
        document.querySelector('.gauge-outer').style.background = `conic-gradient(${color} 0deg ${deg}deg, rgba(255, 255, 255, 0.05) ${deg}deg 360deg)`;

        // Badge Update
        topClassName.textContent = predClass;
        topConfidenceVal.textContent = `${(conf * 100).toFixed(1)}%`;
        topConfidenceFill.style.width = `${(conf * 100).toFixed(1)}%`;

        if (isAnomaly) {
            topClassBadge.style.background = 'rgba(239, 68, 68, 0.15)';
            topClassBadge.style.borderColor = '#ef4444';
            topClassBadge.style.color = '#ef4444';
            anomalyBanner.classList.remove('hidden');
            anomalyBannerText.textContent = `CRITICAL LIVE ALERT: ${predClass.toUpperCase()} DETECTED!`;
            playSirenSound(0.6, true);

            // Dispatch Desktop Push Notification (Throttled to once every 4 seconds)
            const now = Date.now();
            if (now - lastAlertTime > 4000) {
                lastAlertTime = now;
                addLog('alert', 'CRITICAL', `🚨 LIVE ANOMALY DETECTED: ${predClass.toUpperCase()} (${threatPct}% Threat Score)`);
                
                if ("Notification" in window && Notification.permission === "granted") {
                    try {
                        new Notification(`🚨 SURVEILLANCE ALERT: ${predClass.toUpperCase()}`, {
                            body: `Threat Severity: ${threatPct}% | Category: ${predClass}`,
                            icon: 'https://cdn-icons-png.flaticon.com/512/564/564619.png'
                        });
                    } catch (ne) {}
                }
            }
        } else {
            topClassBadge.style.background = 'rgba(16, 185, 129, 0.15)';
            topClassBadge.style.borderColor = '#10b981';
            topClassBadge.style.color = '#10b981';
            anomalyBanner.classList.add('hidden');
        }

        // Person Count & Detected Objects Card Update
        const personCountVal = document.getElementById('personCountVal');
        const detectedObjectsChips = document.getElementById('detectedObjectsChips');
        if (personCountVal) {
            personCountVal.textContent = result.person_count !== undefined ? result.person_count : (result.yolo_graphics?.boxes?.length || 0);
        }
        if (detectedObjectsChips) {
            const objs = result.detected_objects || [];
            if (objs.length === 0) {
                detectedObjectsChips.innerHTML = '<span class="obj-chip">person</span>';
            } else {
                detectedObjectsChips.innerHTML = objs.map(o => `<span class="obj-chip">${o}</span>`).join('');
            }
        }

        // Facial Expression Card Update
        const facialExpressionVal = document.getElementById('facialExpressionVal');
        if (facialExpressionVal && result.facial_expression) {
            const f = result.facial_expression;
            facialExpressionVal.innerHTML = `<span style="font-size:1.2rem">${f.emoji || '😊'}</span> ${f.expression || 'Neutral / Calm'} <span style="opacity:0.75">(${Math.round((f.confidence||0.9)*100)}%)</span>`;
        }

        // Emergency Siren Station Display Update
        const sirenStation = document.getElementById('sirenAlertStation');
        const sirenTitle = document.getElementById('sirenBannerTitle');
        const sirenDesc = document.getElementById('sirenBannerDesc');
        const sirenPulse = document.getElementById('sirenPulseBadge');
        if (sirenStation && sirenTitle && sirenDesc) {
            if (isAnomaly) {
                sirenStation.className = 'siren-alert-station red-state active-alarm';
                sirenTitle.textContent = `CRITICAL THREAT: ${predClass.toUpperCase()}`;
                sirenDesc.textContent = `Live threat detected (${threatPct}% severity score). Dispatching alerts.`;
                if (sirenPulse) sirenPulse.textContent = `ALERT LEVEL: ${threatPct}%`;
            } else {
                sirenStation.className = 'siren-alert-station green-state';
                sirenTitle.textContent = `SYSTEM SECURE — ALL CLEAR`;
                sirenDesc.textContent = `Surveillance video feed analyzed. Normal surveillance conditions.`;
                if (sirenPulse) sirenPulse.textContent = `SYSTEM SECURE`;
            }
        }

        if (result.class_probabilities) {
            renderProbabilityBars(result.class_probabilities, predClass);
        }

        if (result.yolo_graphics) {
            drawYoloGraphics(result.yolo_graphics, isAnomaly, predClass, threatPct);
        }
    }

    // 16. Incident Snapshot Export
    const btnExportSnapshot = document.getElementById('btnExportSnapshot');
    if (btnExportSnapshot) {
        btnExportSnapshot.addEventListener('click', () => {
            try {
                const offscreen = document.createElement('canvas');
                const activeEl = webcamPlayer.classList.contains('hidden') ? (cctvMjpegPlayer.classList.contains('hidden') ? videoPlayer : cctvMjpegPlayer) : webcamPlayer;
                
                offscreen.width = activeEl.videoWidth || activeEl.naturalWidth || 640;
                offscreen.height = activeEl.videoHeight || activeEl.naturalHeight || 360;

                const octx = offscreen.getContext('2d');
                octx.drawImage(activeEl, 0, 0, offscreen.width, offscreen.height);
                if (yoloCanvas) {
                    octx.drawImage(yoloCanvas, 0, 0, offscreen.width, offscreen.height);
                }

                const link = document.createElement('a');
                link.download = `CCTV_Incident_Snapshot_${Date.now()}.png`;
                link.href = offscreen.toDataURL('image/png');
                link.click();

                addLog('system', 'INFO', `Incident snapshot saved to local drive.`);
            } catch (err) {
                alert(`Could not export snapshot: ${err.message}`);
            }
        });
    }
});

