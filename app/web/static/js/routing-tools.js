function showRoutingDryRunModal() {
    document.getElementById('modalContent').innerHTML = '<h2>' + t('routing.dryRunTitle') + '</h2>' +
        '<div class="form-group"><label>' + t('routing.dryRunUser') + '</label>' +
            userSelectHtml('dryRunUsername', '') + '</div>' +
        '<div class="form-group"><label>' + t('routing.dryRunKey') + '</label>' +
            '<input type="text" id="dryRunKey" list="dryRunKeyList" placeholder="sk-...">' +
            '<datalist id="dryRunKeyList"></datalist></div>' +
        '<div class="form-group"><label>' + t('routing.dryRunProvider') + '</label>' +
            providerSelectHtml('dryRunProvider', '', "refreshDryRunModelsForProvider()") + '</div>' +
        '<div class="form-group"><label>' + t('routing.dryRunModel') + '</label>' +
            '<input type="text" id="dryRunModel" list="dryRunModelList" placeholder="provider/model" autocomplete="off">' +
            '<datalist id="dryRunModelList"></datalist></div>' +
        '<div class="form-group"><label>' + t('routing.dryRunResolvedModel') + '</label>' +
            '<input type="text" id="dryRunResolvedModel" list="dryRunModelList" autocomplete="off"></div>' +
        '<div id="dryRunResult" class="dry-run-result" style="display:none;"></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('common.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="runRoutingDryRun()">' + t('routing.dryRunSubmit') + '</button></div>';
    populateDryRunDatalists();
    document.getElementById('modal').style.display = 'flex';
}

function populateDryRunDatalists() {
    refreshDryRunModelsForProvider();
    var keyList = document.getElementById('dryRunKeyList');
    if (keyList) {
        var opts = [];
        var seen = {};
        for (var i = 0; i < users.length; i++) {
            var keys = users[i].api_keys || [];
            for (var j = 0; j < keys.length; j++) {
                var k = keys[j].key;
                if (seen[k]) continue;
                seen[k] = true;
                opts.push('<option value="' + escHtml(k) + '">');
            }
        }
        keyList.innerHTML = opts.join('');
    }
}

function refreshDryRunModelsForProvider() {
    refreshModelDatalistForProvider('dryRunModel', 'dryRunModelList', 'dryRunProvider', false);
    refreshModelDatalistForProvider('dryRunResolvedModel', 'dryRunModelList', 'dryRunProvider', false);
}

async function runRoutingDryRun() {
    var model = document.getElementById('dryRunModel').value.trim();
    if (!model) { toast(t('routing.dryRunNoModel'), 'error'); return; }
    var resultEl = document.getElementById('dryRunResult');
    try {
        var result = await api('/admin/routing-rules/dry-run', {
            method: 'POST',
            body: JSON.stringify({
                username: document.getElementById('dryRunUsername').value.trim(),
                api_key: document.getElementById('dryRunKey').value.trim(),
                model: model,
                resolved_model: document.getElementById('dryRunResolvedModel').value.trim()
            })
        });
        resultEl.innerHTML = routingDryRunResultHtml(result);
        resultEl.style.display = 'block';
    } catch (e) {
        toast(t('routing.dryRunFail') + ': ' + e.message, 'error');
    }
}

function routingDryRunResultHtml(result) {
    var routing = result.routing || {};
    var provider = result.provider || {};
    var effective = result.effective || {};
    var fallback = result.fallback_preview || {};
    var matchedLabel = routing.matched ? t('routing.dryRunMatched') : t('routing.dryRunNoMatch');
    var providerLabel = provider.found
        ? escHtml(provider.id || '-') + (provider.provider_type ? ' (' + escHtml(provider.provider_type) + ')' : '')
        : escHtml(provider.id || '-');
    return '<div class="dry-run-summary">' +
            '<span class="status-dot ' + (routing.matched ? 'on' : 'off') + '"></span>' +
            '<strong>' + matchedLabel + '</strong>' +
        '</div>' +
        '<div class="dry-run-grid">' +
            '<div><span class="label">' + t('routing.name') + '</span><code>' + escHtml(routing.rule_name || routing.rule_id || '-') + '</code></div>' +
            '<div><span class="label">' + t('routing.targetModel') + '</span><code>' + escHtml(routing.target_model || '-') + '</code></div>' +
            '<div><span class="label">' + t('routing.dryRunProvider') + '</span><code>' + providerLabel + '</code></div>' +
            '<div><span class="label">' + t('routing.dryRunEffective') + '</span><code>' + escHtml((effective.provider_id ? effective.provider_id + '/' : '') + (effective.model || '-')) + '</code></div>' +
        '</div>' +
        '<div class="dry-run-reason"><span class="label">' + t('routing.dryRunReason') + '</span> ' + escHtml(routing.reason || '-') + '</div>' +
        '<div class="dry-run-reason"><span class="label">' + t('routing.dryRunFallback') + '</span> ' + routingDryRunFallbackHtml(fallback) + '</div>';
}

function routingDryRunFallbackHtml(fallback) {
    if (!fallback || !fallback.matched) return '<code>-</code>';
    var chain = fallback.chain || [];
    var text = chain.map(function(target) {
        return (target.provider_id ? target.provider_id + '/' : '') + (target.model || '');
    }).join(' -> ');
    return '<code>' + escHtml(fallback.policy_name || fallback.policy_id || '-') + '</code> <span>' + escHtml(text || '-') + '</span>';
}
