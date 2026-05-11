/* ═══════════════════════════════════════════════
   Skin ML Benchmark — JavaScript (Testing Logic)
   ═══════════════════════════════════════════════ */

async function apiCall(url, data) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
    return resp.json();
}

function showSpinner(id) { const el = document.getElementById(id); if (el) el.classList.add('active'); }
function hideSpinner(id) { const el = document.getElementById(id); if (el) el.classList.remove('active'); }

function getBaseFeatures() {
    return {
        Age: document.getElementById('inp-age')?.value || 30,
        Gender: document.getElementById('inp-gender')?.value || 'Female',
        Hydration_Level: document.getElementById('inp-hydration')?.value || 'Medium',
        Oil_Level: document.getElementById('inp-oil')?.value || 'Medium',
        Sensitivity: document.getElementById('inp-sensitivity')?.value || 'Low',
        Humidity: document.getElementById('inp-humidity')?.value || 50,
        Temperature: document.getElementById('inp-temperature')?.value || 20
    };
}

/* ── Classification Test ── */
async function predictClassification() {
    const algo = document.getElementById('clf-algorithm').value;
    const input = getBaseFeatures();
    
    showSpinner('clf-spinner');
    document.getElementById('clf-placeholder').style.display = 'none';
    document.getElementById('clf-results-container').style.display = 'none';

    try {
        const res = await apiCall('/api/classification/predict', { algorithm: algo, input });
        
        document.getElementById('clf-results-container').style.display = 'block';
        document.getElementById('clf-pred-type').textContent = res.skin_type;
        document.getElementById('res-algo-name').textContent = algo.toUpperCase().replace('_', ' ');

        // Probability bars
        const probaList = document.getElementById('clf-pred-proba');
        probaList.innerHTML = '';
        if (res.probabilities) {
            Object.entries(res.probabilities).sort((a,b) => b[1] - a[1]).forEach(([cls, val]) => {
                probaList.innerHTML += `
                    <div class="proba-row">
                        <div class="proba-info"><span>${cls}</span><span>${val}%</span></div>
                        <div class="proba-track"><div class="proba-bar" style="width: ${val}%"></div></div>
                    </div>`;
            });
        }
    } catch (e) {
        alert('Error: ' + e.message);
        document.getElementById('clf-placeholder').style.display = 'flex';
    } finally {
        hideSpinner('clf-spinner');
    }
}

/* ── Regression Test ── */
async function predictRegression() {
    const algo = document.getElementById('reg-algorithm').value;
    const target = document.getElementById('reg-target').value;
    const input = getBaseFeatures();

    showSpinner('reg-spinner');
    document.getElementById('reg-placeholder').style.display = 'none';
    document.getElementById('reg-results-container').style.display = 'none';

    try {
        const res = await apiCall('/api/regression/predict', { algorithm: algo, target: target, input });
        
        document.getElementById('reg-results-container').style.display = 'block';
        document.getElementById('reg-pred-value').textContent = res.prediction;
        document.getElementById('reg-pred-target-label').textContent = target.replace('_', ' ').toUpperCase();
        document.getElementById('res-reg-algo').textContent = algo.toUpperCase();
    } catch (e) {
        alert('Error: ' + e.message);
        document.getElementById('reg-placeholder').style.display = 'flex';
    } finally {
        hideSpinner('reg-spinner');
    }
}

/* ── Clustering Test ── */
async function predictClustering() {
    const algo = document.getElementById('clu-algorithm').value;
    const input = getBaseFeatures();

    showSpinner('clu-spinner');
    document.getElementById('clu-placeholder').style.display = 'none';
    document.getElementById('clu-results-container').style.display = 'none';

    try {
        const res = await apiCall('/api/clustering/predict', { algorithm: algo, input });

        document.getElementById('clu-results-container').style.display = 'block';
        document.getElementById('clu-res-name').textContent = res.cluster_name;
        document.getElementById('clu-res-match').textContent = `via ${res.algorithm}`;

        const details = document.getElementById('clu-group-details');
        if (res.profile) {
            details.innerHTML = `
                <p style="margin-bottom:12px">Based on your features, you belong to <strong>${res.cluster_name}</strong>. This group typically consists of:</p>
                <ul class="advice-list">
                    <li>Dominant Skin Type: <strong>${res.profile.dominant_skin_type}</strong></li>
                    <li>Average Age: <strong>${res.profile.avg_age}</strong></li>
                    <li>Average Health Score: <strong>${res.profile.avg_skin_score}</strong></li>
                </ul>
            `;
        } else {
            details.innerHTML = `<p>Your profile is very unique compared to the rest of the dataset and was flagged as an outlier (Noise) by the DBSCAN algorithm.</p>`;
        }

        // ── Render the cluster scatter plot ──────────────────────────────
        if (res.plot) {
            const plotCard = document.getElementById('clu-plot-card');
            const plotImg  = document.getElementById('clu-plot-img');
            plotImg.src = res.plot;
            plotCard.style.display = 'block';
            // Re-trigger fade-in animation
            plotCard.style.animation = 'none';
            plotCard.offsetHeight; // reflow
            plotCard.style.animation = '';
        }
    } catch (e) {
        alert('Error: ' + e.message);
        document.getElementById('clu-placeholder').style.display = 'flex';
        document.getElementById('clu-plot-card').style.display = 'none';
    } finally {
        hideSpinner('clu-spinner');
    }
}


/* ── Recommendation ── */
async function getRecommendation() {
    const input = getBaseFeatures();
    showSpinner('rec-spinner');
    document.getElementById('rec-placeholder').style.display = 'none';
    try {
        const res = await apiCall('/api/recommendation/get', { input });
        document.getElementById('rec-results').style.display = 'block';
        document.getElementById('rec-skin-type').textContent = res.predicted_skin_type;
        document.getElementById('rec-desc').textContent = res.skincare_advice.description;
        
        document.getElementById('rec-routine').innerHTML = res.skincare_advice.routine.map(s => `<li>${s}</li>`).join('');
        document.getElementById('rec-use').innerHTML = res.skincare_advice.ingredients_to_use.map(i => `<span class="tag use">${i}</span>`).join('');
        document.getElementById('rec-avoid').innerHTML = res.skincare_advice.ingredients_to_avoid.map(i => `<span class="tag avoid">${i}</span>`).join('');

        document.getElementById('rec-similar').innerHTML = res.similar_profiles.map(p => `
            <tr>
                <td>#${p.rank}</td>
                <td>${p.Age}</td>
                <td>${p.Skin_Type}</td>
                <td>${p.skin_score}</td>
                <td style="color:var(--accent-secondary)">${p.similarity}%</td>
            </tr>
        `).join('');
    } catch (e) {
        alert('Error: ' + e.message);
    } finally {
        hideSpinner('rec-spinner');
    }
}

/* ── ANN ── */
async function predictANN() {
    const input  = getBaseFeatures();
    const params = {
        h1:            document.getElementById('ann-h1')?.value || 100,
        h2:            document.getElementById('ann-h2')?.value || 50,
        learning_rate: document.getElementById('ann-lr')?.value || 0.001,
    };

    showSpinner('ann-spinner');
    document.getElementById('ann-placeholder').style.display = 'none';
    document.getElementById('ann-results').style.display     = 'none';

    try {
        const res = await apiCall('/api/ann/predict', { input, params });

        document.getElementById('ann-results').style.display = 'block';
        document.getElementById('ann-pred').textContent      = res.skin_type;
        document.getElementById('ann-meta').textContent      =
            `Val accuracy: ${res.val_accuracy ?? '—'}% · ${res.n_iter} epochs · layers: [${res.hidden_layers.join(', ')}]`;

        // Probability bars
        const probaEl = document.getElementById('ann-proba');
        probaEl.innerHTML = '';
        if (res.probabilities) {
            Object.entries(res.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .forEach(([cls, val]) => {
                probaEl.innerHTML += `
                    <div class="proba-row">
                        <div class="proba-info"><span>${cls}</span><span>${val}%</span></div>
                        <div class="proba-track"><div class="proba-bar" style="width:${val}%"></div></div>
                    </div>`;
            });
        }

        // Plots
        if (res.plot_network)    document.getElementById('ann-plot-network').src    = res.plot_network;
        if (res.plot_importance) document.getElementById('ann-plot-importance').src = res.plot_importance;

    } catch (e) {
        alert('Error: ' + e.message);
        document.getElementById('ann-placeholder').style.display = 'flex';
    } finally {
        hideSpinner('ann-spinner');
    }
}
