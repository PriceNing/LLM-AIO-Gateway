/* 设置页：运行参数（config.json defaults）/ 备份与迁移 / 诊断。
   依赖 app.js 的全局 t / api / escHtml / jsEsc / toast；i18n 键仍集中在 app.js 的 I18N。 */

var settingsState = {
    items: [],
    byKey: {},
    groups: [],
    file: null,
    readOnly: null,
    hooks: [],
    dirty: {},
    loading: false
};

function settingsLoadingBlock() {
    return '<div class="loading-spinner"><span class="spinner"></span><p>' + escHtml(t('stats.loading')) + '</p></div>';
}

function showSettingsTab(tab) {
    ['runtime', 'backup', 'diagnostics'].forEach(function (name) {
        var el = document.getElementById('settings-' + name);
        if (el) el.style.display = (name === tab) ? 'block' : 'none';
    });
    document.querySelectorAll('[data-settings-tab]').forEach(function (btn) {
        btn.classList.toggle('active', btn.getAttribute('data-settings-tab') === tab);
    });
    if (tab === 'diagnostics') renderDiagnostics();
}

async function loadSettings() {
    var container = document.getElementById('settingsRuntimeContent');
    if (container) container.innerHTML = settingsLoadingBlock();
    settingsState.loading = true;
    try {
        var data = await api('/admin/settings');
        settingsState.items = data.items || [];
        settingsState.byKey = {};
        settingsState.items.forEach(function (item) { settingsState.byKey[item.key] = item; });
        settingsState.groups = data.groups || [];
        settingsState.file = data.file || null;
        settingsState.readOnly = data.readOnly || null;
        settingsState.hooks = data.runtimeHooks || [];
        settingsState.dirty = {};
        renderSettingsStatus();
        renderSettingsRuntime();
        renderDiagnostics();
    } catch (e) {
        if (container) container.innerHTML = '<div class="error-text">' + escHtml(e.message) + '</div>';
    } finally {
        settingsState.loading = false;
    }
}

function renderSettingsStatus() {
    var el = document.getElementById('settingsStatusLine');
    if (!el) return;
    var file = settingsState.file || {};
    var parts = ['<span class="mono">' + escHtml(file.path || '-') + '</span>'];
    if (file.exists === false) {
        parts.push('<span class="badge badge-cancelled">' + escHtml(t('settings.fileAbsent')) + '</span>');
    }
    parts.push(file.writable
        ? '<span class="badge badge-ok">' + escHtml(t('settings.writable')) + '</span>'
        : '<span class="badge badge-fail">' + escHtml(t('settings.notWritable')) + '</span>');
    if (file.mtime) {
        parts.push('<span class="mono">' + escHtml(t('settings.mtime')) + ' ' + escHtml(new Date(file.mtime * 1000).toLocaleString()) + '</span>');
    }
    parts.push('<span class="muted">' + escHtml(t('settings.hotAll')) + '</span>');
    el.innerHTML = parts.join(' ');
}

/* -- 数值格式化：bytes / seconds 这类原始数字人眼不可读，必须换算显示 -- */

function settingsNumber(value) {
    var num = Number(value);
    if (!isFinite(num)) return String(value);
    return num.toLocaleString('en-US');
}

function settingsFormatBytes(value) {
    var num = Number(value);
    if (!isFinite(num) || num <= 0) return settingsNumber(value);
    var units = ['B', 'KiB', 'MiB', 'GiB'];
    var size = num;
    var index = 0;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    var text = (index === 0 || size >= 100) ? Math.round(size).toString() : size.toFixed(1);
    return text + ' ' + units[index] + ' (' + settingsNumber(num) + ' B)';
}

function settingsFormatSeconds(value) {
    var num = Number(value);
    if (!isFinite(num)) return String(value);
    if (num >= 86400) return (num / 86400).toFixed(1) + ' d (' + settingsNumber(num) + ' s)';
    if (num >= 3600) return (num / 3600).toFixed(1) + ' h (' + settingsNumber(num) + ' s)';
    if (num >= 120) return (num / 60).toFixed(1) + ' min (' + settingsNumber(num) + ' s)';
    return settingsNumber(num) + ' s';
}

function settingsDisplayValue(item, value) {
    if (value === true) return t('settings.boolTrue');
    if (value === false) return t('settings.boolFalse');
    if (value === void 0 || value === null || value === '') return '-';
    if (Array.isArray(value)) return value.join(', ');
    var unit = item ? item.unit : '';
    if (unit === 'bytes') return settingsFormatBytes(value);
    if (unit === 'seconds') return settingsFormatSeconds(value);
    if (unit === 'tokens' || unit === 'count' || unit === 'pixels') return settingsNumber(value);
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

/* -- 风险本地预检：服务端仍会重算，这里只是让管理员在提交前就看到警告 -- */

function settingsDangerHit(item, value) {
    var rule = item ? item.danger_rule : null;
    if (!rule || !rule.when) return null;
    var hit = false;
    if (rule.when === 'true') {
        hit = (value === true);
    } else if (rule.when === 'false') {
        hit = (value === false);
    } else if (rule.when === 'empty') {
        hit = Array.isArray(value)
            ? value.length === 0
            : (typeof value === 'string' ? value.length === 0 : false);
    } else if (rule.when === 'lte' || rule.when === 'gte') {
        var left = Number(value);
        var right = Number(rule.value);
        if (isFinite(left) && isFinite(right)) {
            hit = (rule.when === 'lte') ? (left <= right) : (left >= right);
        }
    }
    return hit ? (rule.key || 'settings.danger') : null;
}

/* -- dirty 值优先：界面显示与提交用同一个取值口径，避免"看着改了、提交的是旧值" -- */

function settingsRawValue(item) {
    if (Object.prototype.hasOwnProperty.call(settingsState.dirty, item.key)) {
        return settingsState.dirty[item.key];
    }
    return item.value;
}

function settingsScalar(value) {
    if (Array.isArray(value)) {
        return value.join('|');
    }
    if (value === null || value === void 0) {
        return '';
    }
    if (typeof value === 'object') {
        return JSON.stringify(value);
    }
    return String(value);
}

function settingsValuesEqual(a, b) {
    if (Object.is(a, b)) {
        return true;
    }
    if (settingsScalar(a) === settingsScalar(b)) {
        return true;
    }
    if (String(a).trim() === '' || String(b).trim() === '') {
        return false;
    }
    var na = Number(a);
    var nb = Number(b);
    return isFinite(na) && isFinite(nb) && na === nb;
}

/* -- 表单解析 + 本地范围校验：服务端仍是权威，这里只把明显错值挡在提交前 -- */

function settingsParseInput(item, el) {
    var type = item.type;
    if (type === 'bool') {
        return String(el.value) === 'true';
    }
    var text = String(el.value).trim();
    if (type === 'int' || type === 'float') {
        if (text === '') {
            toast(item.key + ': ' + t('settings.needNumber'), 'error');
            return null;
        }
        var num = Number(text);
        if (!isFinite(num)) {
            toast(item.key + ': ' + t('settings.needNumber'), 'error');
            return null;
        }
        if (type === 'int' && num !== Math.floor(num)) {
            toast(item.key + ': ' + t('settings.needInt'), 'error');
            return null;
        }
        if (typeof item.min === 'number' && num < item.min) {
            toast(t('settings.rangeMin', { key: item.key, min: item.min }), 'error');
            return null;
        }
        if (typeof item.max === 'number' && num > item.max) {
            toast(t('settings.rangeMax', { key: item.key, max: item.max }), 'error');
            return null;
        }
        return num;
    }
    if (type === 'string_list') {
        var list = [];
        text.split(',').forEach(function (part) {
            var one = part.trim();
            if (one !== '' && list.indexOf(one) === -1) {
                list.push(one);
            }
        });
        return list;
    }
    if (type === 'json_object') {
        if (text === '') {
            return {};
        }
        try {
            var parsed = JSON.parse(text);
            if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
                toast(item.key + ': ' + t('settings.badJson'), 'error');
                return null;
            }
            return parsed;
        } catch (err) {
            toast(item.key + ': ' + t('settings.badJson'), 'error');
            return null;
        }
    }
    if (type === 'url' && text === '') {
        toast(item.key + ': ' + t('settings.needValue'), 'error');
        return null;
    }
    return text;
}

/* -- 改动只存在界面内存里，点"保存"才落盘；随时可放弃 -- */

function settingsIsDirty(key) {
    return Object.prototype.hasOwnProperty.call(settingsState.dirty, key);
}

function onSettingChange(key, el) {
    var item = settingsState.byKey[key];
    if (!item) {
        return;
    }
    var parsed = settingsParseInput(item, el);
    if (parsed === null) {
        return;
    }
    if (settingsValuesEqual(parsed, item.value)) {
        delete settingsState.dirty[key];
    } else {
        settingsState.dirty[key] = parsed;
    }
    refreshSettingsRow(key);
}

function settingsInputValue(item, value) {
    if (Array.isArray(value)) {
        return value.join(', ');
    }
    if (value === null || value === void 0) {
        return '';
    }
    if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
    }
    return String(value);
}

/* -- 控件：id 固定为 setting-<key>，重渲染与保存都按这个 id 找元素 -- */

function settingsControl(item, value) {
    var attrs = ' id="setting-' + escHtml(item.key) + '"' +
        ' data-setting-key="' + escHtml(item.key) + '"' +
        ' onchange="onSettingChange(\'' + jsEsc(item.key) + '\', this)"';
    var type = item.type;
    if (type === 'bool') {
        var isOn = (value === true);
        return '<select class="select-input"' + attrs + '>' +
            '<option value="true"' + (isOn ? ' selected' : '') + '>' + escHtml(t('settings.boolTrue')) + '</option>' +
            '<option value="false"' + (isOn ? '' : ' selected') + '>' + escHtml(t('settings.boolFalse')) + '</option>' +
            '</select>';
    }
    if (type === 'int' || type === 'float') {
        var range = '';
        if (typeof item.min === 'number') { range += ' min="' + escHtml(item.min) + '"'; }
        if (typeof item.max === 'number') { range += ' max="' + escHtml(item.max) + '"'; }
        var step = (type === 'int') ? '1' : 'any';
        return '<input type="number" class="text-input num-input" step="' + step + '"' + range +
            ' value="' + escHtml(settingsInputValue(item, value)) + '"' + attrs + '>';
    }
    if (type === 'json_object') {
        return '<textarea class="text-input json-input" rows="4"' + attrs + '>' +
            escHtml(settingsInputValue(item, value)) + '</textarea>';
    }
    return '<input type="text" class="text-input wide" value="' +
        escHtml(settingsInputValue(item, value)) + '"' + attrs + '>';
}

/* -- 行渲染：key / 提示 / 风险 / 控件 / 元信息（默认值、来源、重置） -- */

function settingsRow(item) {
    var cls = 'settings-row';
    if (settingsIsDirty(item.key)) { cls += ' dirty'; }
    if (settingsDangerHit(item, settingsRawValue(item))) { cls += ' danger'; }
    return '<div class="' + cls + '" id="settings-row-' + escHtml(item.key) + '">' +
        settingsRowInner(item) + '</div>';
}

function settingsRowInner(item) {
    var value = settingsRawValue(item);
    var dangerKey = settingsDangerHit(item, value);
    var parts = [];
    parts.push('<div class="settings-main">');
    parts.push('<div class="settings-label"><code>' + escHtml(item.key) + '</code>' +
        (settingsIsDirty(item.key)
            ? '<span class="badge badge-partial">' + escHtml(t('settings.dirtyTag')) + '</span>'
            : '') + '</div>');
    if (item.hint) {
        parts.push('<div class="settings-hint">' + escHtml(t(item.hint)) + '</div>');
    }
    if (dangerKey) {
        parts.push('<div class="settings-danger">' + escHtml(t(dangerKey)) + '</div>');
    }
    parts.push('</div>');
    parts.push('<div class="settings-control">' + settingsControl(item, value) + '</div>');
    parts.push('<div class="settings-meta">' + settingsMeta(item, value) + '</div>');
    return parts.join('');
}

function settingsMeta(item, value) {
    var parts = [];
    parts.push('<span class="settings-kv"><em>' + escHtml(t('settings.default')) + '</em> ' +
        escHtml(settingsDisplayValue(item, item.default)) + '</span>');
    if (!settingsValuesEqual(item.value, item.default)) {
        parts.push('<span class="settings-kv"><em>' + escHtml(t('settings.current')) + '</em> ' +
            escHtml(settingsDisplayValue(item, item.value)) + '</span>');
    }
    if (settingsIsDirty(item.key)) {
        parts.push('<span class="settings-kv"><em>' + escHtml(t('settings.pendingValue')) + '</em> ' +
            escHtml(settingsDisplayValue(item, value)) + '</span>');
    }
    var fromFile = (item.source === 'file');
    parts.push('<span class="badge ' + (fromFile ? 'badge-endpoint' : 'badge-cancelled') + '">' +
        escHtml(fromFile ? t('settings.sourceFile') : t('settings.sourceBuiltin')) + '</span>');
    if (item.written) {
        parts.push('<button class="btn btn-secondary btn-sm" onclick="resetSetting(\'' +
            jsEsc(item.key) + '\')">' + escHtml(t('settings.reset')) + '</button>');
    }
    return parts.join(' ');
}

/* -- 分组渲染：schema 的 groups 顺序即展示顺序；未归组的键落到末尾"其它"卡片，绝不隐藏 -- */

function renderSettingsRuntime() {
    var container = document.getElementById('settingsRuntimeContent');
    if (!container) {
        return;
    }
    var groups = settingsState.groups || [];
    var buckets = {};
    var order = [];
    groups.forEach(function (group) {
        buckets[group.id] = [];
        order.push({ id: group.id, label: group.label || 'settings.group.other' });
    });
    var other = [];
    (settingsState.items || []).forEach(function (item) {
        if (Object.prototype.hasOwnProperty.call(buckets, item.group)) {
            buckets[item.group].push(item);
        } else {
            other.push(item);
        }
    });
    if (other.length) {
        buckets['__other__'] = other;
        order.push({ id: '__other__', label: 'settings.group.other' });
    }
    var html = order.map(function (group) {
        var items = buckets[group.id] || [];
        if (!items.length) {
            return '';
        }
        return '<div class="settings-card glass"><h3>' + escHtml(t(group.label)) + '</h3>' +
            items.map(function (item) { return settingsRow(item); }).join('') + '</div>';
    }).join('');
    container.innerHTML = html + renderSettingsSaveBar();
}

/* -- 保存条：待保存条数 + 风险确认；无改动时保存按钮 disabled -- */

function settingsPendingDangers() {
    var out = [];
    Object.keys(settingsState.dirty).forEach(function (key) {
        var item = settingsState.byKey[key];
        if (!item) {
            return;
        }
        var dangerKey = settingsDangerHit(item, settingsState.dirty[key]);
        if (dangerKey) {
            out.push({ key: key, value: settingsState.dirty[key], dangerKey: dangerKey });
        }
    });
    return out;
}

function settingsSaveBarBody() {
    var keys = Object.keys(settingsState.dirty);
    var dangers = settingsPendingDangers();
    var parts = [];
    if (keys.length) {
        parts.push('<span class="settings-pending">' + escHtml(t('settings.pending', { n: keys.length })) + '</span>');
    } else {
        parts.push('<span class="muted">' + escHtml(t('settings.saveBarEmpty')) + '</span>');
    }
    if (dangers.length) {
        var labels = dangers.map(function (d) {
            return d.key + '=' + settingsScalar(d.value) + ' (' + t(d.dangerKey) + ')';
        }).join('; ');
        parts.push('<label class="checkbox settings-danger"><input type="checkbox" id="settingsConfirmDanger"> ' +
            '<span>' + escHtml(t('settings.confirmDanger') + ': ' + labels) + '</span></label>');
    }
    parts.push('<span class="settings-spacer"></span>');
    parts.push('<button class="btn btn-secondary" onclick="discardSettings()">' + escHtml(t('settings.discard')) + '</button>');
    parts.push('<button class="btn btn-primary" onclick="saveSettings()"' +
        (keys.length ? '' : ' disabled') + '>' + escHtml(t('settings.save')) + '</button>');
    return parts.join(' ');
}

function renderSettingsSaveBar() {
    var active = Object.keys(settingsState.dirty).length ? ' active' : '';
    return '<div class="settings-savebar' + active + '" id="settingsSaveBar">' + settingsSaveBarBody() + '</div>';
}

function refreshSettingsSaveBar() {
    var el = document.getElementById('settingsSaveBar');
    if (!el) {
        renderSettingsRuntime();
        return;
    }
    el.classList.toggle('active', Object.keys(settingsState.dirty).length > 0);
    el.innerHTML = settingsSaveBarBody();
}

function refreshSettingsRow(key) {
    var row = document.getElementById('settings-row-' + key);
    var item = settingsState.byKey[key];
    if (!row || !item) {
        renderSettingsRuntime();
        return;
    }
    var cls = 'settings-row';
    if (settingsIsDirty(key)) { cls += ' dirty'; }
    if (settingsDangerHit(item, settingsRawValue(item))) { cls += ' danger'; }
    row.className = cls;
    row.innerHTML = settingsRowInner(item);
    refreshSettingsSaveBar();
}

function discardSettings() {
    settingsState.dirty = {};
    renderSettingsRuntime();
}

/* -- 提交：一次批量 PUT；成功后用服务端回包覆盖本地状态，不做乐观更新 -- */

function settingsApplyItems(data) {
    if (data && data.items) {
        settingsState.items = data.items;
        settingsState.byKey = {};
        settingsState.items.forEach(function (item) { settingsState.byKey[item.key] = item; });
    }
    if (data && data.file) {
        settingsState.file = data.file;
    }
    settingsState.dirty = {};
    renderSettingsStatus();
    renderSettingsRuntime();
    renderDiagnostics();
}

async function saveSettings() {
    var keys = Object.keys(settingsState.dirty);
    if (!keys.length) {
        return;
    }
    var values = {};
    keys.forEach(function (key) { values[key] = settingsState.dirty[key]; });
    var dangers = settingsPendingDangers();
    var box = document.getElementById('settingsConfirmDanger');
    var confirmed = !!(box && box.checked);
    if (dangers.length && !confirmed) {
        toast(t('settings.dangerConfirmRequired'), 'error');
        return;
    }
    try {
        var data = await api('/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({ values: values, confirmDanger: confirmed })
        });
        settingsApplyItems(data);
        toast(t('settings.saved', { n: keys.length }), 'success');
        var failedNames = Object.keys(data.failedHooks || {});
        if (failedNames.length) {
            toast(t('settings.hookFailed') + ': ' + failedNames.join(', '), 'warning');
        }
    } catch (e) {
        toast(t('settings.saveFail') + ': ' + e.message, 'error');
    }
}

/* -- 重置为默认 = 从文件删除该键：写回当前默认值会把值永久钉死在文件里 -- */

async function resetSetting(key) {
    if (!confirm(t('settings.confirmReset'))) {
        return;
    }
    try {
        var data = await api('/admin/settings/reset', {
            method: 'POST',
            body: JSON.stringify({ keys: [key] })
        });
        settingsApplyItems(data);
        toast(t('settings.resetDone'), 'success');
    } catch (e) {
        toast(t('settings.resetFail') + ': ' + e.message, 'error');
    }
}

/* -- 从磁盘重读 config.json：给"SSH 手改文件"这条工作流用，改完不必重启网关 -- */

async function reloadConfigFile() {
    try {
        var data = await api('/admin/config/reload', { method: 'POST' });
        var changed = data.changedKeys || [];
        toast(t('settings.reloadDone', { n: changed.length }), 'success');
        var restart = data.restartRequiredChangedKeys || [];
        if (restart.length) {
            toast(t('settings.restartNeeded') + ': ' + restart.join(', '), 'warning');
        }
    } catch (e) {
        toast(t('settings.reloadFail') + ': ' + e.message, 'error');
    }
    await loadSettings();
}

/* -- 诊断：版本 / 配置文件状态 / 只读顶层项 / 运行时钩子 / 输出预算关系 -- */

function settingsPlainValue(value) {
    if (value === null || value === void 0) {
        return '-';
    }
    if (typeof value === 'boolean') {
        return value ? t('settings.boolTrue') : t('settings.boolFalse');
    }
    if (typeof value === 'object') {
        return JSON.stringify(value);
    }
    return String(value);
}

function settingsKv(label, valueHtml) {
    return '<div class="settings-kv"><em>' + escHtml(label) + '</em> <span>' + valueHtml + '</span></div>';
}

function renderDiagnostics() {
    var el = document.getElementById('settingsDiagnosticsContent');
    if (!el) {
        return;
    }
    var file = settingsState.file || {};
    var ro = settingsState.readOnly || {};
    var version = (typeof serviceVersion === 'string' && serviceVersion) ? serviceVersion : '-';
    var parts = [];
    parts.push('<div class="settings-card glass"><h3>' + escHtml(t('settings.diag.service')) + '</h3>');
    parts.push(settingsKv(t('settings.diag.version'), '<span class="mono">' + escHtml(version) + '</span>'));
    parts.push(settingsKv(t('settings.diag.file'), '<span class="mono">' + escHtml(file.path || '-') + '</span>'));
    parts.push(settingsKv(t('settings.diag.exists'), file.exists
        ? '<span class="badge badge-ok">' + escHtml(t('settings.filePresent')) + '</span>'
        : '<span class="badge badge-cancelled">' + escHtml(t('settings.fileAbsent')) + '</span>'));
    parts.push(settingsKv(t('settings.diag.writable'), file.writable
        ? '<span class="badge badge-ok">' + escHtml(t('settings.writable')) + '</span>'
        : '<span class="badge badge-fail">' + escHtml(t('settings.notWritable')) + '</span>'));
    if (file.mtime) {
        parts.push(settingsKv(t('settings.mtime'), '<span class="mono">' +
            escHtml(new Date(file.mtime * 1000).toLocaleString()) + '</span>'));
    }
    parts.push('</div>');
    var readOnlyKeys = ['host', 'port', 'reload', 'database', 'image_result_dir', 'cors_allow_origins', 'logging'];
    parts.push('<div class="settings-card glass"><h3>' + escHtml(t('settings.diag.readOnly')) + '</h3>');
    parts.push('<p class="muted">' + escHtml(t('settings.readOnlyHint')) + '</p>');
    readOnlyKeys.forEach(function (key) {
        parts.push('<div class="settings-kv"><em>' + escHtml(key) + '</em> ' +
            '<span class="mono">' + escHtml(settingsPlainValue(ro[key])) + '</span> ' +
            '<span class="badge badge-cancelled">' + escHtml(t('settings.restartRequired')) + '</span></div>');
    });
    parts.push('</div>');
    var hooks = settingsState.hooks || [];
    parts.push('<div class="settings-card glass"><h3>' + escHtml(t('settings.diag.hooks')) + '</h3>');
    parts.push('<p class="muted">' + escHtml(t('settings.hooksHint')) + '</p>');
    parts.push(hooks.length
        ? hooks.map(function (name) { return '<span class="badge badge-ok">' + escHtml(name) + '</span>'; }).join(' ')
        : '<span class="muted">-</span>');
    parts.push('</div>');
    parts.push('<div class="settings-card glass"><h3>' + escHtml(t('settings.budgetNoteTitle')) + '</h3>' +
        '<p class="muted">' + escHtml(t('settings.budgetNote')) + '</p></div>');
    el.innerHTML = parts.join('');
}
