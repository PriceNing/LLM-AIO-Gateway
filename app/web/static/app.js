/* ═══════════════════════════════════════════════════════════════════════
   LLM AIO Gateway Admin — Application
   ═══════════════════════════════════════════════════════════════════════ */

const API_BASE = '';
const SESSION_KEY = 'llm_gateway_admin_session';
const LANG_KEY = 'llm_gateway_lang';
const THEME_KEY = 'llm_gateway_theme';

let authMode = 'login';
let providers = [];
let models = [];
let allModels = [];
let users = [];
let currentLang = localStorage.getItem(LANG_KEY) || 'zh';

/* ═══════════════════════════════ i18n ═══════════════════════════════ */

const I18N = {
zh: {
    'auth.title': 'LLM AIO Gateway',
    'auth.hint': '登录后管理用户、模型和调用 Key。',
    'auth.hintSetup': '首次使用，请创建第一个管理员账号。',
    'auth.username': '管理员账号',
    'auth.password': '密码',
    'auth.login': '登录',
    'auth.create': '创建管理员',
    'auth.fail': '登录失败',
    'auth.initFail': '初始化失败',
    'auth.emptyFields': '请输入账号和密码',

    'nav.users': '用户管理',
    'nav.providers': '提供商',
    'nav.models': '模型管理',
    'nav.routing': '路由规则',
    'nav.stats': '统计',
    'nav.preprocessors': '视觉模型注入',
    'nav.logout': '退出',
    'nav.switchLang': '切换语言',
    'nav.switchTheme': '切换主题',
    'nav.changePassword': '修改密码',

    'password.title': '修改密码',
    'password.current': '当前密码',
    'password.new': '新密码',
    'password.confirm': '确认新密码',
    'password.submit': '修改',
    'password.success': '密码修改成功',
    'password.mismatch': '两次输入的新密码不一致',
    'password.tooShort': '新密码不能少于6位',
    'password.wrongCurrent': '当前密码错误',

    'users.title': '用户管理',
    'users.add': '新增用户',
    'users.empty': '暂无用户。创建用户后，为其生成调用 API Key 并授权模型。',
    'users.enabled': '启用',
    'users.disabled': '禁用',
    'users.allModels': '全部模型',
    'users.wildcard': '通配符 *',
    'users.modelsHint': '模型列表加载中，请先切换到"模型管理"页面加载数据...',
    'users.addTitle': '新增用户',
    'users.editTitle': '编辑用户',
    'users.username': '用户名',
    'users.displayName': '显示名称',
    'users.allowedModels': '允许模型',
    'users.save': '保存',
    'users.cancel': '取消',
    'users.addFail': '新增用户失败',
    'users.updateFail': '更新用户失败',
    'users.deleteConfirm': '确定要删除这个用户吗？',
    'users.deleteFail': '删除用户失败',
    'users.keyTitle': '生成调用 API Key',
    'users.keyName': '名称',
    'users.keyGenerate': '生成',
    'users.keyFail': '生成 Key 失败',
    'users.keyCopied': 'API Key 已生成并复制到剪贴板。',
    'users.keyDeleteConfirm': '确定要删除这个 API Key 吗？',
    'users.keyDeleteFail': '删除 Key 失败',
    'users.keyEdit': '编辑',
    'users.keyEditTitle': '编辑 API Key',
    'users.keyUpdateFail': '更新 Key 失败',
    'users.calls': '调用',
    'users.failed': '失败',
    'users.tokens': 'Tokens',

    'providers.title': '提供商列表',
    'providers.add': '新增提供商',
    'providers.empty': '暂无配置的提供商',
    'providers.enabled': '启用',
    'providers.disabled': '禁用',
    'providers.modelsCount': '{n} 个模型',
    'providers.edit': '编辑',
    'providers.refresh': '刷新',
    'providers.delete': '删除',
    'providers.addTitle': '新增提供商',
    'providers.editTitle': '编辑提供商',
    'providers.id': 'ID',
    'providers.idPlaceholder': '唯一标识，例如 openai-main',
    'providers.idHint': '仅允许字母、数字、点、下划线、连字符，禁止空格',
    'providers.name': '名称',
    'providers.type': '类型',
    'providers.typeOpenAI': 'OpenAI 兼容',
    'providers.typeAnthropic': 'Anthropic 兼容',
    'providers.apiBase': 'API Base URL',
    'providers.apiKey': '上游 API Key',
    'providers.addFail': '新增失败',
    'providers.updateFail': '更新失败',
    'providers.deleteConfirm': '确定要删除这个提供商吗？',
    'providers.deleteFail': '删除失败',
    'providers.refreshOk': '刷新成功，发现 {n} 个模型',
    'providers.refreshFail': '刷新失败',
    'providers.refreshAllDone': '刷新完成',
    'providers.refreshAllFail': '刷新失败',

    'models.title': '模型列表',
    'models.empty': '暂无模型',
    'models.search': '搜索模型...',
    'models.refreshAll': '刷新所有模型',
    'models.copyId': '复制 ID',
    'models.count': '个模型',
    'models.loadFail': '加载模型失败',

    'routing.title': '路由规则',
    'routing.add': '新增规则',
    'routing.empty': '暂无规则。创建规则后，根据条件将请求路由到指定模型。',
    'routing.addTitle': '新增路由规则',
    'routing.editTitle': '编辑路由规则',
    'routing.name': '规则名称',
    'routing.username': '匹配用户（空=全部）',
    'routing.keyPattern': '匹配 Key（空=全部）',
    'routing.matchModel': '匹配请求模型',
    'routing.matchModelHint': '支持 * 通配符，如 deepseek-*',
    'routing.targetModel': '目标模型',
    'routing.targetProvider': '目标提供商（空=自动）',
    'routing.save': '保存',
    'routing.cancel': '取消',
    'routing.loadFail': '加载路由规则失败',
    'routing.noMatchModel': '请填写匹配请求模型',
    'routing.addFail': '新增规则失败',
    'routing.updateFail': '更新规则失败',
    'routing.deleteConfirm': '确定要删除这条规则吗？',
    'routing.deleteFail': '删除规则失败',
    'routing.enabled': '启用',
    'routing.disabled': '禁用',

    'stats.title': '调用统计',
    'stats.loadFail': '加载统计失败',
    'stats.totalCalls': '总调用次数',
    'stats.successRate': '成功率',
    'stats.failedCalls': '失败次数',
    'stats.activeModels': '活跃模型',
    'stats.noData': '暂无调用数据',
    'stats.noDataHint': '通过 API 发送请求后，统计数据将在此显示',
    'stats.reset': '上次重置',
    'stats.autoRefresh': '每5秒自动刷新',
    'stats.resetBtn': '清空统计数据',
    'stats.resetConfirm': '确定要清空所有统计数据吗？\n\n此操作将重置：\n- 全局调用计数\n- 所有用户/API Key 用量统计\n- 实时请求日志\n\n此操作不可撤销。',
    'stats.resetFail': '清空失败',
    'stats.realtime': '实时请求日志',
    'stats.time': '时间',
    'stats.client': '客户端',
    'stats.key': 'Key',
    'stats.model': '实际模型',
    'stats.requestedModel': '请求模型',
    'stats.endpoint': '端点',
    'stats.tokens': 'Tokens',
    'stats.status': '状态',
    'stats.noRecords': '暂无记录',
    'stats.modelDist': '模型用量分布',
    'stats.timeline': '请求时间线',
    'stats.chartSuccess': '成功',
    'stats.chartFail': '失败',
    'stats.loadUsersFail': '加载用户失败',
    'stats.loadProvidersFail': '加载提供商失败',

    'preprocessors.title': '视觉模型注入',
    'preprocessors.add': '新增预处理器',
    'preprocessors.empty': '暂无配置的预处理器',
    'preprocessors.configTitle': '预处理器配置',
    'preprocessors.addTitle': '新增预处理器',
    'preprocessors.editTitle': '编辑预处理器',
    'preprocessors.name': '名称',
    'preprocessors.namePlaceholder': '例如 vision-model',
    'preprocessors.nameRequired': '请输入预处理器名称',
    'preprocessors.apiBase': 'API Base URL',
    'preprocessors.apiBasePlaceholder': '例如 http://localhost:8001',
    'preprocessors.model': '模型名称',
    'preprocessors.modelPlaceholder': '例如 Qwen-VL',
    'preprocessors.apiKey': 'API Key',
    'preprocessors.apiKeyPlaceholder': '视觉模型 API Key（可选）',
    'preprocessors.timeout': '超时时间（秒）',
    'preprocessors.maxImages': '最大图片数',
    'preprocessors.maxTokens': '最大 Token 数',
    'preprocessors.prompt': '图像描述提示词',
    'preprocessors.enabled': '启用',
    'preprocessors.disabled': '禁用',
    'preprocessors.save': '保存',
    'preprocessors.cancel': '取消',
    'preprocessors.delete': '删除',
    'preprocessors.deleteConfirm': '确定要删除这个预处理器吗？',
    'preprocessors.deleteFail': '删除预处理器失败',
    'preprocessors.addFail': '新增预处理器失败',
    'preprocessors.updateFail': '更新预处理器失败',
    'preprocessors.loadFail': '加载预处理器失败',
    'preprocessors.modelsTitle': '模型开关',
    'preprocessors.modelsEmpty': '暂无模型数据',
    'preprocessors.modelsOn': '开',
    'preprocessors.modelsOff': '关',
    'preprocessors.toggleFail': '切换失败',
    'preprocessors.fetchModels': '获取模型',
    'preprocessors.needApiBase': '请先填写 API Base URL',
    'preprocessors.modelsFound': '个模型已获取',
    'preprocessors.noModels': '未发现可用模型',
    'preprocessors.fetchFail': '获取模型列表失败',

    'common.save': '保存',
    'common.cancel': '取消',
    'common.copy': '复制',
    'common.copied': '已复制',
    'common.copy_failed': '复制失败，请手动复制',
    'common.delete': '删除',
    'common.edit': '编辑',
    'common.close': '关闭',
    'common.yes': '确定',
    'common.no': '取消',
},

en: {
    'auth.title': 'LLM AIO Gateway',
    'auth.hint': 'Log in to manage users, models, and API keys.',
    'auth.hintSetup': 'First time — create the initial admin account.',
    'auth.username': 'Admin Username',
    'auth.password': 'Password',
    'auth.login': 'Log In',
    'auth.create': 'Create Admin',
    'auth.fail': 'Login failed',
    'auth.initFail': 'Initialization failed',
    'auth.emptyFields': 'Please enter username and password',

    'nav.users': 'Users',
    'nav.providers': 'Providers',
    'nav.models': 'Models',
    'nav.routing': 'Routing',
    'nav.stats': 'Stats',
    'nav.preprocessors': 'Vision Model Injection',
    'nav.logout': 'Logout',
    'nav.switchLang': 'Switch Language',
    'nav.switchTheme': 'Toggle Theme',
    'nav.changePassword': 'Change Password',

    'password.title': 'Change Password',
    'password.current': 'Current Password',
    'password.new': 'New Password',
    'password.confirm': 'Confirm New Password',
    'password.submit': 'Change',
    'password.success': 'Password changed successfully',
    'password.mismatch': 'Passwords do not match',
    'password.tooShort': 'Password must be at least 6 characters',
    'password.wrongCurrent': 'Current password is incorrect',

    'users.title': 'User Management',
    'users.add': 'Add User',
    'users.empty': 'No users yet. Create a user, then generate API keys with model access.',
    'users.enabled': 'Enabled',
    'users.disabled': 'Disabled',
    'users.allModels': 'All Models',
    'users.wildcard': 'wildcard *',
    'users.modelsHint': 'Loading models… switch to Models tab first to load data.',
    'users.addTitle': 'Add User',
    'users.editTitle': 'Edit User',
    'users.username': 'Username',
    'users.displayName': 'Display Name',
    'users.allowedModels': 'Allowed Models',
    'users.save': 'Save',
    'users.cancel': 'Cancel',
    'users.addFail': 'Failed to add user',
    'users.updateFail': 'Failed to update user',
    'users.deleteConfirm': 'Delete this user?',
    'users.deleteFail': 'Failed to delete user',
    'users.keyTitle': 'Generate API Key',
    'users.keyName': 'Name',
    'users.keyGenerate': 'Generate',
    'users.keyFail': 'Failed to generate key',
    'users.keyCopied': 'API key generated and copied to clipboard.',
    'users.keyDeleteConfirm': 'Delete this API key?',
    'users.keyDeleteFail': 'Failed to delete key',
    'users.keyEdit': 'Edit',
    'users.keyEditTitle': 'Edit API Key',
    'users.keyUpdateFail': 'Failed to update key',
    'users.calls': 'calls',
    'users.failed': 'failed',
    'users.tokens': 'tokens',

    'providers.title': 'Providers',
    'providers.add': 'Add Provider',
    'providers.empty': 'No providers configured',
    'providers.enabled': 'Enabled',
    'providers.disabled': 'Disabled',
    'providers.modelsCount': '{n} models',
    'providers.edit': 'Edit',
    'providers.refresh': 'Refresh',
    'providers.delete': 'Delete',
    'providers.addTitle': 'Add Provider',
    'providers.editTitle': 'Edit Provider',
    'providers.id': 'ID',
    'providers.idPlaceholder': 'Unique ID, e.g. openai-main',
    'providers.idHint': 'Only letters, digits, dots, underscores, hyphens. No spaces.',
    'providers.name': 'Name',
    'providers.type': 'Type',
    'providers.typeOpenAI': 'OpenAI Compatible',
    'providers.typeAnthropic': 'Anthropic Compatible',
    'providers.apiBase': 'API Base URL',
    'providers.apiKey': 'Upstream API Key',
    'providers.addFail': 'Failed to add',
    'providers.updateFail': 'Failed to update',
    'providers.deleteConfirm': 'Delete this provider?',
    'providers.deleteFail': 'Failed to delete',
    'providers.refreshOk': 'Refresh OK — {n} models found',
    'providers.refreshFail': 'Refresh failed',
    'providers.refreshAllDone': 'Refresh complete',
    'providers.refreshAllFail': 'Refresh failed',

    'models.title': 'Models',
    'models.empty': 'No models found',
    'models.search': 'Search models...',
    'models.refreshAll': 'Refresh All',
    'models.copyId': 'Copy ID',
    'models.count': 'models',
    'models.loadFail': 'Failed to load models',

    'routing.title': 'Routing Rules',
    'routing.add': 'Add Rule',
    'routing.empty': 'No rules. Create a rule to route requests based on conditions.',
    'routing.addTitle': 'Add Routing Rule',
    'routing.editTitle': 'Edit Routing Rule',
    'routing.name': 'Rule Name',
    'routing.username': 'Match User (empty=all)',
    'routing.keyPattern': 'Match Key (empty=all)',
    'routing.matchModel': 'Match Request Model',
    'routing.matchModelHint': 'Supports * wildcard, e.g. deepseek-*',
    'routing.targetModel': 'Target Model',
    'routing.targetProvider': 'Target Provider (empty=auto)',
    'routing.save': 'Save',
    'routing.cancel': 'Cancel',
    'routing.loadFail': 'Failed to load routing rules',
    'routing.noMatchModel': 'Match model is required',
    'routing.addFail': 'Failed to add rule',
    'routing.updateFail': 'Failed to update rule',
    'routing.deleteConfirm': 'Delete this rule?',
    'routing.deleteFail': 'Failed to delete rule',
    'routing.enabled': 'Enabled',
    'routing.disabled': 'Disabled',

    'stats.title': 'Statistics',
    'stats.loadFail': 'Failed to load stats',
    'stats.totalCalls': 'Total Calls',
    'stats.successRate': 'Success Rate',
    'stats.failedCalls': 'Failed Calls',
    'stats.activeModels': 'Active Models',
    'stats.noData': 'No data yet',
    'stats.noDataHint': 'Send requests through the API and stats will appear here.',
    'stats.reset': 'Last reset',
    'stats.autoRefresh': 'Auto-refresh every 5s',
    'stats.resetBtn': 'Clear Statistics',
    'stats.resetConfirm': 'Clear all statistics?\n\nThis will reset:\n- Global call counters\n- All user/API key usage\n- Request log\n\nThis cannot be undone.',
    'stats.resetFail': 'Reset failed',
    'stats.realtime': 'Real-time Request Log',
    'stats.time': 'Time',
    'stats.client': 'Client',
    'stats.key': 'Key',
    'stats.model': 'Actual Model',
    'stats.requestedModel': 'Requested',
    'stats.endpoint': 'Endpoint',
    'stats.tokens': 'Tokens',
    'stats.status': 'Status',
    'stats.noRecords': 'No records',
    'stats.modelDist': 'Model Distribution',
    'stats.timeline': 'Request Timeline',
    'stats.chartSuccess': 'Success',
    'stats.chartFail': 'Failed',
    'stats.loadUsersFail': 'Failed to load users',
    'stats.loadProvidersFail': 'Failed to load providers',

    'preprocessors.title': 'Vision Model Injection',
    'preprocessors.add': 'Add Preprocessor',
    'preprocessors.empty': 'No preprocessors configured',
    'preprocessors.configTitle': 'Preprocessor Configuration',
    'preprocessors.addTitle': 'Add Preprocessor',
    'preprocessors.editTitle': 'Edit Preprocessor',
    'preprocessors.name': 'Name',
    'preprocessors.namePlaceholder': 'e.g. vision-model',
    'preprocessors.nameRequired': 'Preprocessor name is required',
    'preprocessors.apiBase': 'API Base URL',
    'preprocessors.apiBasePlaceholder': 'e.g. http://localhost:8001',
    'preprocessors.model': 'Model Name',
    'preprocessors.modelPlaceholder': 'e.g. Qwen-VL',
    'preprocessors.apiKey': 'API Key',
    'preprocessors.apiKeyPlaceholder': 'Vision model API Key (optional)',
    'preprocessors.timeout': 'Timeout (seconds)',
    'preprocessors.maxImages': 'Max Images',
    'preprocessors.maxTokens': 'Max Tokens',
    'preprocessors.prompt': 'Image Description Prompt',
    'preprocessors.enabled': 'Enabled',
    'preprocessors.disabled': 'Disabled',
    'preprocessors.save': 'Save',
    'preprocessors.cancel': 'Cancel',
    'preprocessors.delete': 'Delete',
    'preprocessors.deleteConfirm': 'Delete this preprocessor?',
    'preprocessors.deleteFail': 'Failed to delete preprocessor',
    'preprocessors.addFail': 'Failed to add preprocessor',
    'preprocessors.updateFail': 'Failed to update preprocessor',
    'preprocessors.loadFail': 'Failed to load preprocessors',
    'preprocessors.modelsTitle': 'Model Toggles',
    'preprocessors.modelsEmpty': 'No models available',
    'preprocessors.modelsOn': 'ON',
    'preprocessors.modelsOff': 'OFF',
    'preprocessors.toggleFail': 'Toggle failed',
    'preprocessors.fetchModels': 'Fetch Models',
    'preprocessors.needApiBase': 'Please enter API Base URL first',
    'preprocessors.modelsFound': 'models found',
    'preprocessors.noModels': 'No models found',
    'preprocessors.fetchFail': 'Failed to fetch models',

    'common.save': 'Save',
    'common.cancel': 'Cancel',
    'common.copy': 'Copy',
    'common.copied': 'Copied',
    'common.copy_failed': 'Copy failed, please copy manually',
    'common.delete': 'Delete',
    'common.edit': 'Edit',
    'common.close': 'Close',
    'common.yes': 'Yes',
    'common.no': 'No',
}
};

function t(key, params) {
    let s = (I18N[currentLang] && I18N[currentLang][key]) || (I18N['zh'][key]) || key;
    if (params) {
        for (const [k, v] of Object.entries(params)) {
            s = s.replace('{'+k+'}', v);
        }
    }
    return s;
}

function applyI18n() {
    document.documentElement.lang = currentLang === 'en' ? 'en' : 'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (el.tagName === 'INPUT' && el.placeholder !== undefined && el.hasAttribute('data-i18n-placeholder')) {
            el.placeholder = t(key);
        } else if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') {
            // skip value attributes — just handle placeholders
        } else {
            el.textContent = t(key);
        }
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        el.title = t(el.getAttribute('data-i18n-title'));
    });
}

function toggleLang() {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    localStorage.setItem(LANG_KEY, currentLang);
    applyI18n();
    // Re-render visible section
    const visible = document.querySelector('.section[style*="block"], .section:not([style])');
    if (visible) {
        const id = visible.id.replace('-section', '');
        refreshSection(id);
    }
    updateAuthHint();
}

function refreshSection(section) {
    if (section === 'users') { renderUsers(); }
    if (section === 'providers') renderProviders();
    if (section === 'models') renderModels();
    if (section === 'stats') loadStats();
}

/* ═══════════════════════════════ Theme ═══════════════════════════════ */

function initTheme() {
    const saved = localStorage.getItem(THEME_KEY) || 'dark';
    applyTheme(saved);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    const btn = document.getElementById('btnTheme');
    if (btn) btn.innerHTML = theme === 'dark' ? '&#9788;' : '&#9790;';
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
    // Re-render charts if on stats page
    const statsSection = document.getElementById('stats-section');
    if (statsSection && statsSection.style.display !== 'none') {
        loadStats();
    }
}

/* ═══════════════════════════════ Toast ═══════════════════════════════ */

function toast(msg, type) {
    type = type || 'info';
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(function() {
        el.classList.add('removing');
        setTimeout(function() { el.remove(); }, 300);
    }, 3000);
}

/* ═══════════════════════════════ Auth ═══════════════════════════════ */

function getToken() {
    return localStorage.getItem(SESSION_KEY) || '';
}

function getHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    const token = getToken();
    if (token) headers.Authorization = 'Bearer ' + token;
    return headers;
}

async function api(path, options) {
    options = options || {};
    const res = await fetch(API_BASE + path, Object.assign({}, options, {
        headers: Object.assign({}, getHeaders(), options.headers || {})
    }));
    const data = await res.json().catch(function() { return {}; });
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    return data;
}

function updateAuthHint() {
    document.getElementById('authHint').textContent = authMode === 'setup'
        ? t('auth.hintSetup')
        : t('auth.hint');
}

async function initAuth() {
    try {
        var status = await api('/auth/status');
        authMode = status.has_admin ? 'login' : 'setup';
        updateAuthHint();
        document.querySelector('.auth-submit').textContent = status.has_admin ? t('auth.login') : t('auth.create');

        if (getToken()) {
            try {
                var me = await api('/auth/me');
                enterApp(me);
            } catch (e) {
                localStorage.removeItem(SESSION_KEY);
            }
        }
    } catch (err) {
        toast(t('auth.initFail') + ': ' + err.message, 'error');
    }
}

async function submitAuth() {
    var username = document.getElementById('adminUsername').value.trim();
    var password = document.getElementById('adminPassword').value;
    if (!username || !password) {
        toast(t('auth.emptyFields'), 'error');
        return;
    }
    try {
        var path = authMode === 'setup' ? '/auth/setup' : '/auth/login';
        var data = await api(path, {
            method: 'POST',
            body: JSON.stringify({ username: username, password: password, display_name: username })
        });
        localStorage.setItem(SESSION_KEY, data.token);
        enterApp(data);
    } catch (e) {
        toast(t('auth.fail') + ': ' + e.message, 'error');
    }
}

function enterApp(admin) {
    document.getElementById('authView').style.display = 'none';
    document.getElementById('appView').style.display = 'block';
    document.getElementById('currentAdmin').textContent = admin.display_name || admin.username;
    loadAll();
}

async function logout() {
    try { await api('/auth/logout', { method: 'POST' }); } catch (e) {}
    localStorage.removeItem(SESSION_KEY);
    location.reload();
}

function showChangePasswordModal() {
    document.getElementById('modalContent').innerHTML =
        '<h3 data-i18n="password.title">' + t('password.title') + '</h3>' +
        '<div class="form-group"><label data-i18n="password.current">' + t('password.current') + '</label>' +
        '<input type="password" id="currentPwd" autocomplete="current-password"></div>' +
        '<div class="form-group"><label data-i18n="password.new">' + t('password.new') + '</label>' +
        '<input type="password" id="newPwd1" autocomplete="new-password"></div>' +
        '<div class="form-group"><label data-i18n="password.confirm">' + t('password.confirm') + '</label>' +
        '<input type="password" id="newPwd2" autocomplete="new-password" onkeydown="if(event.key===\'Enter\')submitChangePassword()"></div>' +
        '<div style="display:flex;gap:1rem;margin-top:1.5rem;">' +
        '<button class="btn btn-secondary" onclick="closeModal()" data-i18n="users.cancel">' + t('users.cancel') + '</button>' +
        '<button class="btn btn-primary" onclick="submitChangePassword()" data-i18n="password.submit">' + t('password.submit') + '</button>' +
        '</div>';
    document.getElementById('modal').style.display = 'flex';
}

async function submitChangePassword() {
    var current = document.getElementById('currentPwd').value;
    var pw1 = document.getElementById('newPwd1').value;
    var pw2 = document.getElementById('newPwd2').value;
    if (!current || !pw1) return;
    if (pw1.length < 6) { toast(t('password.tooShort'), 'error'); return; }
    if (pw1 !== pw2) { toast(t('password.mismatch'), 'error'); return; }
    try {
        var resp = await api('/auth/password', {
            method: 'PUT',
            body: JSON.stringify({ current_password: current, new_password: pw1 })
        });
        if (resp.status === 'ok') {
            toast(t('password.success'), 'success');
            closeModal();
        }
    } catch (e) {
        toast(t('password.wrongCurrent'), 'error');
    }
}

async function loadAll() {
    await Promise.all([loadModels(), loadUsers(), loadProviders()]);
}

/* ═══════════════════════════════ Navigation ═══════════════════════════════ */

function showSection(section, evt) {
    document.querySelectorAll('.section').forEach(function(s) { s.style.display = 'none'; });
    document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });

    var target = document.getElementById(section + '-section');
    if (target) target.style.display = 'block';

    var btn = document.querySelector('[data-section="' + section + '"]');
    if (btn) btn.classList.add('active');

    stopStatsTimer();
    if (section === 'users') { loadUsers(); loadModels(); }
    if (section === 'providers') loadProviders();
    if (section === 'models') loadModels();
    if (section === 'routing') { loadRoutingRules(); loadUsers(); loadModels(); loadProviders(); }
    if (section === 'stats') loadStats();
    if (section === 'preprocessors') loadPreprocessors();
}

/* ═══════════════════════════════ Users ═══════════════════════════════ */

async function loadUsers() {
    try {
        var data = await api('/admin/users');
        users = data.users || [];
        renderUsers();
    } catch (e) {
        toast(t('stats.loadUsersFail') + ': ' + e.message, 'error');
    }
}

function renderUsers() {
    var container = document.getElementById('usersList');
    if (!users.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#128101;</div><p>' + t('users.empty') + '</p></div>';
        return;
    }

    container.innerHTML = users.map(function(user) {
        var calls = (user.stats && user.stats.total_calls) || 0;
        var failed = (user.stats && user.stats.failed_calls) || 0;
        var tokens = (user.stats && user.stats.total_tokens) || 0;
        return '<div class="user-card glass">' +
            '<div class="user-top">' +
                '<div class="user-info">' +
                    '<div class="user-name">' + escHtml(user.display_name || user.username) + '</div>' +
                    '<div class="user-meta">' + escHtml(user.username) + ' / ' + (user.enabled === false ? t('users.disabled') : t('users.enabled')) + '</div>' +
                '</div>' +
                '<div class="user-actions">' +
                    '<button class="btn btn-secondary btn-sm" onclick="showEditUserModal(\'' + jsEsc(user.username) + '\')">' + t('common.edit') + '</button>' +
                    '<button class="btn btn-primary btn-sm" onclick="showAddUserKeyModal(\'' + jsEsc(user.username) + '\')">' + t('users.keyGenerate') + ' Key</button>' +
                    '<button class="btn btn-danger btn-sm" onclick="deleteUser(\'' + jsEsc(user.username) + '\')">' + t('common.delete') + '</button>' +
                '</div>' +
            '</div>' +
            '<div class="user-detail">' +
                '<span>' + t('users.calls') + ': ' + calls.toLocaleString() + '</span>' +
                '<span>' + t('users.failed') + ': ' + failed.toLocaleString() + '</span>' +
                '<span>' + t('users.tokens') + ': ' + tokens.toLocaleString() + '</span>' +
            '</div>' +
            '<div class="key-list">' +
                (user.api_keys || []).map(function(key) {
                    return '<div class="key-row">' +
                        '<span><strong>' + escHtml(key.name) + '</strong> ' + fmtModels(key.allowed_models) + '</span>' +
                        '<code>' + escHtml(maskKey(key.key)) + '</code>' +
                        '<span class="key-stats">' + t('users.calls') + ' ' + ((key.stats && key.stats.total_calls) || 0).toLocaleString() + '</span>' +
                        '<button class="btn btn-secondary btn-xs" data-editkey="' + encodeURIComponent(JSON.stringify({u: user.username, k: key.key, n: key.name, m: key.allowed_models || []})) + '" onclick="editUserKeyFromBtn(this)">' + t('users.keyEdit') + '</button>' +
                        '<button class="btn btn-secondary btn-xs" onclick="copyText(\'' + jsEsc(key.key) + '\')">' + t('common.copy') + '</button>' +
                        '<button class="btn btn-danger btn-xs" onclick="deleteUserKey(\'' + jsEsc(user.username) + '\',\'' + jsEsc(key.key) + '\')">' + t('common.delete') + '</button>' +
                    '</div>';
                }).join('') +
            '</div>' +
        '</div>';
    }).join('');
}

function showAddUserModal() {
    document.getElementById('modalContent').innerHTML = userFormHtml(t('users.addTitle'), {}, 'addUser()');
    document.getElementById('modal').style.display = 'flex';
}

function showEditUserModal(username) {
    var user = users.find(function(item) { return item.username === username; });
    if (!user) return;
    document.getElementById('modalContent').innerHTML = userFormHtml(t('users.editTitle'), user, 'updateUser(\'' + jsEsc(username) + '\')');
    document.getElementById('modal').style.display = 'flex';
}

function userFormHtml(title, user, action) {
    return '<h2>' + title + '</h2>' +
        '<div class="form-group"><label>' + t('users.username') + '</label>' +
            '<input type="text" id="userUsername" value="' + escHtml(user.username || '') + '"' + (user.username ? ' disabled' : '') + '></div>' +
        '<div class="form-group"><label>' + t('users.displayName') + '</label>' +
            '<input type="text" id="userDisplayName" value="' + escHtml(user.display_name || '') + '"></div>' +
        '<div class="form-group"><label><input type="checkbox" id="userEnabled"' + (user.enabled === false ? '' : ' checked') + '> ' + t('users.enabled') + '</label></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('users.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="' + action + '">' + t('users.save') + '</button></div>';
}

function readUserForm() {
    return {
        username: document.getElementById('userUsername').value.trim(),
        display_name: document.getElementById('userDisplayName').value.trim(),
        enabled: document.getElementById('userEnabled').checked
    };
}

async function addUser() {
    try {
        await api('/admin/users', { method: 'POST', body: JSON.stringify(readUserForm()) });
        closeModal();
        loadUsers();
    } catch (e) { toast(t('users.addFail') + ': ' + e.message, 'error'); }
}

async function updateUser(username) {
    try {
        await api('/admin/users/' + encodeURIComponent(username), { method: 'PUT', body: JSON.stringify(readUserForm()) });
        closeModal();
        loadUsers();
    } catch (e) { toast(t('users.updateFail') + ': ' + e.message, 'error'); }
}

async function deleteUser(username) {
    if (!confirm(t('users.deleteConfirm'))) return;
    try {
        await api('/admin/users/' + encodeURIComponent(username), { method: 'DELETE' });
        loadUsers();
    } catch (e) { toast(t('users.deleteFail') + ': ' + e.message, 'error'); }
}

function showAddUserKeyModal(username) {
    document.getElementById('modalContent').innerHTML =
        '<h2>' + t('users.keyTitle') + '</h2>' +
        '<div class="form-group"><label>' + t('users.keyName') + '</label>' +
            '<input type="text" id="keyName" value="default"></div>' +
        '<div class="form-group"><label>' + t('users.allowedModels') + '</label>' +
            modelSelectorHtml(['*'], 'key') + '</div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('users.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="addUserKey(\'' + jsEsc(username) + '\')">' + t('users.keyGenerate') + '</button></div>';
    document.getElementById('modal').style.display = 'flex';
}

async function addUserKey(username) {
    try {
        var key = await api('/admin/users/' + encodeURIComponent(username) + '/api-keys', {
            method: 'POST',
            body: JSON.stringify({
                name: document.getElementById('keyName').value.trim(),
                allowed_models: readModelSelector('key')
            })
        });
        closeModal();
        await loadUsers();
        await copyText(key.key);
    } catch (e) { toast(t('users.keyFail') + ': ' + e.message, 'error'); }
}

async function deleteUserKey(username, key) {
    if (!confirm(t('users.keyDeleteConfirm'))) return;
    try {
        await api('/admin/users/' + encodeURIComponent(username) + '/api-keys/' + encodeURIComponent(key), { method: 'DELETE' });
        loadUsers();
    } catch (e) { toast(t('users.keyDeleteFail') + ': ' + e.message, 'error'); }
}

function showEditUserKeyModal(username, keyValue, keyName, allowedModels) {
    document.getElementById('modalContent').innerHTML =
        '<h2>' + t('users.keyEditTitle') + '</h2>' +
        '<div class="form-group"><label>' + t('users.keyName') + '</label>' +
            '<input type="text" id="editKeyName" value="' + escHtml(keyName) + '"></div>' +
        '<div class="form-group"><label>' + t('users.allowedModels') + '</label>' +
            modelSelectorHtml(allowedModels || [], 'editKey') + '</div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('users.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="updateUserKey(\'' + jsEsc(username) + '\',\'' + jsEsc(keyValue) + '\')">' + t('users.save') + '</button></div>';
    document.getElementById('modal').style.display = 'flex';
}

async function updateUserKey(username, keyValue) {
    try {
        await api('/admin/users/' + encodeURIComponent(username) + '/api-keys/' + encodeURIComponent(keyValue), {
            method: 'PUT',
            body: JSON.stringify({
                name: document.getElementById('editKeyName').value.trim(),
                allowed_models: readModelSelector('editKey')
            })
        });
        closeModal();
        loadUsers();
    } catch (e) { toast(t('users.keyUpdateFail') + ': ' + e.message, 'error'); }
}

// Bridge: data-attribute stores URL-encoded JSON to avoid quoting issues in HTML
function editUserKeyFromBtn(btn) {
    var d = JSON.parse(decodeURIComponent(btn.getAttribute('data-editkey')));
    showEditUserKeyModal(d.u, d.k, d.n, d.m);
}

/* ═══════════════════════════════ Routing Rules ═══════════════════════════════ */

var routingRules = [];

async function loadRoutingRules() {
    try {
        var data = await api('/admin/routing-rules');
        routingRules = data.rules || [];
        renderRoutingRules();
    } catch (e) {
        toast(t('routing.loadFail') + ': ' + e.message, 'error');
    }
}

function renderRoutingRules() {
    var container = document.getElementById('routingRulesList');
    if (!routingRules.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#128257;</div><p>' + t('routing.empty') + '</p></div>';
        return;
    }
    container.innerHTML = routingRules.map(function(rule) {
        return '<div class="routing-card glass">' +
            '<div class="card-top">' +
                '<div>' +
                    '<div class="card-title">' + escHtml(rule.name) + '</div>' +
                    '<div class="card-meta" style="margin-top:4px;">' +
                        '<span class="status-dot ' + (rule.enabled ? 'on' : 'off') + '"></span>' +
                        '<span>' + (rule.enabled ? t('routing.enabled') : t('routing.disabled')) + '</span>' +
                    '</div>' +
                '</div>' +
                '<div class="card-actions">' +
                    '<button class="btn btn-secondary btn-sm" onclick="showEditRoutingRuleModal(\'' + jsEsc(rule.id) + '\')">' + t('common.edit') + '</button>' +
                    '<button class="btn btn-danger btn-sm" onclick="deleteRoutingRule(\'' + jsEsc(rule.id) + '\')">' + t('common.delete') + '</button>' +
                '</div>' +
            '</div>' +
            '<div class="routing-rule-detail">' +
                '<div class="routing-arrow">' +
                    '<div class="routing-from">' +
                        '<span class="label">' + t('routing.matchModel') + ':</span>' +
                        '<code>' + escHtml(rule.match_model || '*') + '</code>' +
                        (rule.username ? '<span class="label" style="margin-left:8px">User:</span><code>' + escHtml(rule.username) + '</code>' : '') +
                    '</div>' +
                    '<span class="arrow">→</span>' +
                    '<div class="routing-to">' +
                        '<span class="label">' + t('routing.targetModel') + ':</span>' +
                        '<code>' + escHtml(rule.target_model || '-') + '</code>' +
                        (rule.target_provider ? ' @ <code>' + escHtml(rule.target_provider) + '</code>' : '') +
                    '</div>' +
                '</div>' +
            '</div>' +
        '</div>';
    }).join('');
}

function modelSelectHtml(id, selected, includeWildcard) {
    var models = (allModels || []).map(function(m) { return m.id; });
    var opts = includeWildcard ? '<option value="*">* (' + t('routing.matchModelHint') + ')</option>' : '<option value="">--</option>';
    var found = selected === '*' || selected === '';
    for (var i = 0; i < models.length; i++) {
        var sel = models[i] === selected ? ' selected' : '';
        if (models[i] === selected) found = true;
        opts += '<option value="' + escHtml(models[i]) + '"' + sel + '>' + escHtml(models[i]) + '</option>';
    }
    // Preserve custom values (e.g. wildcards) when editing
    if (selected && !found) {
        opts += '<option value="' + escHtml(selected) + '" selected>' + escHtml(selected) + '</option>';
    }
    return '<select id="' + id + '" style="width:100%">' + opts + '</select>';
}

function providerSelectHtml(id, selected) {
    var provs = (providers || []).map(function(p) { return p.id; });
    var opts = '<option value="">' + t('routing.targetProvider') + '</option>';
    for (var i = 0; i < provs.length; i++) {
        var sel = provs[i] === selected ? ' selected' : '';
        opts += '<option value="' + escHtml(provs[i]) + '"' + sel + '>' + escHtml(provs[i]) + '</option>';
    }
    return '<select id="' + id + '" style="width:100%">' + opts + '</select>';
}

function userSelectHtml(id, selected) {
    var opts = '<option value="">' + t('routing.username') + '</option>';
    for (var i = 0; i < users.length; i++) {
        var u = users[i].username;
        var sel = u === selected ? ' selected' : '';
        opts += '<option value="' + escHtml(u) + '"' + sel + '>' + escHtml(u) + '</option>';
    }
    return '<select id="' + id + '" style="width:100%">' + opts + '</select>';
}

function keySelectHtml(id, selected) {
    var opts = '<option value="">' + t('routing.keyPattern') + '</option>';
    var seen = {};
    for (var i = 0; i < users.length; i++) {
        var keys = users[i].api_keys || [];
        for (var j = 0; j < keys.length; j++) {
            var k = keys[j].key;
            if (seen[k]) continue;
            seen[k] = true;
            var label = keys[j].name + ' (' + maskKey(k) + ') - ' + users[i].username;
            var sel = k === selected ? ' selected' : '';
            opts += '<option value="' + escHtml(k) + '"' + sel + '>' + escHtml(label) + '</option>';
        }
    }
    return '<select id="' + id + '" style="width:100%">' + opts + '</select>';
}

function routingFormHtml(title, rule) {
    rule = rule || {};
    return '<h2>' + title + '</h2>' +
        '<div class="form-group"><label>' + t('routing.name') + '</label>' +
            '<input type="text" id="ruleName" value="' + escHtml(rule.name || '') + '"></div>' +
        '<div class="form-group"><label><input type="checkbox" id="ruleEnabled"' + (rule.enabled === false ? '' : ' checked') + '> ' + t('routing.enabled') + '</label></div>' +
        '<div class="form-group"><label>' + t('routing.username') + '</label>' +
            userSelectHtml('ruleUsername', rule.username || '') + '</div>' +
        '<div class="form-group"><label>' + t('routing.keyPattern') + '</label>' +
            keySelectHtml('ruleKeyPattern', rule.api_key_pattern || '') + '</div>' +
        '<div class="form-group"><label>' + t('routing.matchModel') + ' (' + t('routing.matchModelHint') + ')</label>' +
            '<input type="text" id="ruleMatchModel" list="matchModelList" value="' + escHtml(rule.match_model || '') + '" placeholder="*">' +
            '<datalist id="matchModelList"></datalist></div>' +
        '<div class="form-group"><label>' + t('routing.targetModel') + '</label>' +
            modelSelectHtml('ruleTargetModel', rule.target_model || '', false) + '</div>' +
        '<div class="form-group"><label>' + t('routing.targetProvider') + '</label>' +
            providerSelectHtml('ruleTargetProvider', rule.target_provider || '') + '</div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('routing.cancel') + '</button>' +
            '<button class="btn btn-primary" id="ruleSaveBtn">' + t('routing.save') + '</button></div>';
}

function readRoutingForm() {
    return {
        name: document.getElementById('ruleName').value.trim(),
        enabled: document.getElementById('ruleEnabled').checked,
        username: document.getElementById('ruleUsername').value.trim(),
        api_key_pattern: document.getElementById('ruleKeyPattern').value.trim(),
        match_model: document.getElementById('ruleMatchModel').value.trim(),
        target_model: document.getElementById('ruleTargetModel').value.trim(),
        target_provider: document.getElementById('ruleTargetProvider').value.trim()
    };
}

function showAddRoutingRuleModal() {
    document.getElementById('modalContent').innerHTML = routingFormHtml(t('routing.addTitle'));
    populateMatchModelDatalist();
    document.getElementById('ruleSaveBtn').onclick = addRoutingRule;
    document.getElementById('modal').style.display = 'flex';
}

function populateMatchModelDatalist() {
    var dl = document.getElementById('matchModelList');
    if (!dl) return;
    var models = (allModels || []).map(function(m) { return m.id; });
    var opts = '<option value="*">';
    for (var i = 0; i < models.length; i++) {
        opts += '<option value="' + escHtml(models[i]) + '">';
    }
    dl.innerHTML = opts;
}

async function addRoutingRule() {
    var form = readRoutingForm();
    if (!form.match_model) { toast(t('routing.noMatchModel'), 'error'); return; }
    try {
        await api('/admin/routing-rules', { method: 'POST', body: JSON.stringify(form) });
        closeModal();
        loadRoutingRules();
    } catch (e) { toast(t('routing.addFail') + ': ' + e.message, 'error'); }
}

function showEditRoutingRuleModal(ruleId) {
    var rule = routingRules.find(function(r) { return r.id === ruleId; });
    if (!rule) return;
    document.getElementById('modalContent').innerHTML = routingFormHtml(t('routing.editTitle'), rule);
    populateMatchModelDatalist();
    document.getElementById('ruleSaveBtn').onclick = function() { updateRoutingRule(ruleId); };
    document.getElementById('modal').style.display = 'flex';
}

async function updateRoutingRule(ruleId) {
    var form = readRoutingForm();
    if (!form.match_model) { toast(t('routing.noMatchModel'), 'error'); return; }
    try {
        await api('/admin/routing-rules/' + encodeURIComponent(ruleId), {
            method: 'PUT', body: JSON.stringify(form)
        });
        closeModal();
        loadRoutingRules();
    } catch (e) { toast(t('routing.updateFail') + ': ' + e.message, 'error'); }
}

async function deleteRoutingRule(ruleId) {
    if (!confirm(t('routing.deleteConfirm'))) return;
    try {
        await api('/admin/routing-rules/' + encodeURIComponent(ruleId), { method: 'DELETE' });
        loadRoutingRules();
    } catch (e) { toast(t('routing.deleteFail') + ': ' + e.message, 'error'); }
}

/* ═══════════════════════════════ Providers ═══════════════════════════════ */

async function loadProviders() {
    try {
        providers = await api('/admin/providers');
        renderProviders();
    } catch (e) {
        toast(t('stats.loadProvidersFail') + ': ' + e.message, 'error');
    }
}

function renderProviders() {
    var container = document.getElementById('providersList');
    if (!providers.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#9881;</div><p>' + t('providers.empty') + '</p></div>';
        return;
    }

    container.innerHTML = providers.map(function(p) {
        var modelCount = (p.models && p.models.length) || 0;
        return '<div class="provider-card glass glass-hover">' +
            '<div class="card-top">' +
                '<div>' +
                    '<div class="provider-name">' + escHtml(p.name) + '</div>' +
                    '<div class="card-meta" style="margin-top:4px;">' +
                        '<span class="status-dot ' + (p.enabled ? 'on' : 'off') + '"></span>' +
                        '<span>' + (p.enabled ? t('providers.enabled') : t('providers.disabled')) + '</span>' +
                        '<span>' + t('providers.modelsCount', {n: modelCount}) + '</span>' +
                    '</div>' +
                '</div>' +
                '<span class="provider-type">' + escHtml(p.provider_type) + '</span>' +
            '</div>' +
            '<div class="card-actions">' +
                '<button class="btn btn-secondary btn-sm" onclick="editProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.edit') + '</button>' +
                '<button class="btn btn-secondary btn-sm" onclick="refreshProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.refresh') + '</button>' +
                '<button class="btn btn-danger btn-sm" onclick="deleteProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.delete') + '</button>' +
            '</div>' +
        '</div>';
    }).join('');
}

function showAddProviderModal() {
    document.getElementById('modalContent').innerHTML = providerFormHtml(t('providers.addTitle'), {}, 'addProvider()');
    document.getElementById('modal').style.display = 'flex';
}

function providerFormHtml(title, provider, submitAction) {
    return '<h2>' + title + '</h2>' +
        '<div class="form-group"><label>' + t('providers.id') + '</label>' +
            '<input type="text" id="providerId" value="' + escHtml(provider.id || '') + '"' + (provider.id ? ' disabled' : '') + ' placeholder="' + t('providers.idPlaceholder') + '" pattern="[a-zA-Z0-9._-]+" title="' + t('providers.idHint') + '" oninput="this.value=this.value.replace(/\\s/g,\'\')"></div>' +
        '<div class="form-group"><label>' + t('providers.name') + '</label>' +
            '<input type="text" id="providerName" value="' + escHtml(provider.name || '') + '"></div>' +
        '<div class="form-group"><label>' + t('providers.type') + '</label>' +
            '<select id="providerType">' +
                '<option value="openai"' + (provider.provider_type === 'openai' ? ' selected' : '') + '>' + t('providers.typeOpenAI') + '</option>' +
                '<option value="anthropic"' + (provider.provider_type === 'anthropic' ? ' selected' : '') + '>' + t('providers.typeAnthropic') + '</option>' +
            '</select></div>' +
        '<div class="form-group"><label>' + t('providers.apiBase') + '</label>' +
            '<input type="text" id="providerApiBase" value="' + escHtml(provider.api_base || '') + '" placeholder="https://api.openai.com/v1"></div>' +
        '<div class="form-group"><label>' + t('providers.apiKey') + '</label>' +
            '<input type="password" id="providerApiKey" value="' + escHtml(provider.api_key || '') + '"></div>' +
        '<div class="form-group"><label><input type="checkbox" id="providerEnabled"' + (provider.enabled === false ? '' : ' checked') + '> ' + t('providers.enabled') + '</label></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('common.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="' + submitAction + '">' + t('common.save') + '</button></div>';
}

function readProviderForm() {
    return {
        id: document.getElementById('providerId').value.trim(),
        name: document.getElementById('providerName').value.trim(),
        provider_type: document.getElementById('providerType').value,
        api_base: document.getElementById('providerApiBase').value.trim(),
        api_key: document.getElementById('providerApiKey').value.trim(),
        enabled: document.getElementById('providerEnabled').checked
    };
}

async function addProvider() {
    try {
        await api('/admin/providers', { method: 'POST', body: JSON.stringify(Object.assign({}, readProviderForm(), { models: [] })) });
        closeModal();
        loadProviders();
    } catch (e) { toast(t('providers.addFail') + ': ' + e.message, 'error'); }
}

function editProvider(id) {
    var provider = providers.find(function(item) { return item.id === id; });
    if (!provider) return;
    document.getElementById('modalContent').innerHTML = providerFormHtml(t('providers.editTitle'), provider, 'updateProvider(\'' + jsEsc(id) + '\')');
    document.getElementById('modal').style.display = 'flex';
}

async function updateProvider(id) {
    try {
        await api('/admin/providers/' + encodeURIComponent(id), { method: 'PUT', body: JSON.stringify(readProviderForm()) });
        closeModal();
        loadProviders();
    } catch (e) { toast(t('providers.updateFail') + ': ' + e.message, 'error'); }
}

async function deleteProvider(id) {
    if (!confirm(t('providers.deleteConfirm'))) return;
    try {
        await api('/admin/providers/' + encodeURIComponent(id), { method: 'DELETE' });
        loadProviders();
    } catch (e) { toast(t('providers.deleteFail') + ': ' + e.message, 'error'); }
}

async function refreshProvider(id) {
    try {
        var result = await api('/admin/providers/' + encodeURIComponent(id) + '/refresh', { method: 'POST' });
        if (result.error) throw new Error(result.error);
        toast(t('providers.refreshOk', {n: result.count}), 'success');
        await Promise.all([loadProviders(), loadModels()]);
    } catch (e) { toast(t('providers.refreshFail') + ': ' + e.message, 'error'); }
}

async function refreshAllModels() {
    try {
        await api('/admin/providers/refresh-all', { method: 'POST' });
        await Promise.all([loadProviders(), loadModels()]);
        toast(t('providers.refreshAllDone'), 'success');
    } catch (e) { toast(t('providers.refreshAllFail') + ': ' + e.message, 'error'); }
}

/* ═══════════════════════════════ Models ═══════════════════════════════ */

async function loadModels() {
    try {
        var data = await api('/admin/models');
        allModels = data.models || [];
        filterModels();
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

function filterModels() {
    var searchInput = document.getElementById('modelSearch');
    var search = searchInput ? searchInput.value.toLowerCase() : '';
    models = allModels.filter(function(m) {
        return m.id.toLowerCase().indexOf(search) !== -1 ||
            (m.name && m.name.toLowerCase().indexOf(search) !== -1) ||
            m.provider_name.toLowerCase().indexOf(search) !== -1;
    });
    renderModels();
}

function renderModels() {
    var container = document.getElementById('modelsList');
    if (!models.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#128269;</div><p>' + t('models.empty') + '</p></div>';
        return;
    }

    // 按 provider_name 分组
    var grouped = {};
    for (var i = 0; i < models.length; i++) {
        var m = models[i];
        var pname = m.provider_name || m.provider;
        if (!grouped[pname]) grouped[pname] = [];
        grouped[pname].push(m);
    }

    var html = '';
    var providerKeys = Object.keys(grouped).sort();
    for (var pi = 0; pi < providerKeys.length; pi++) {
        var pname = providerKeys[pi];
        var groupModels = grouped[pname];
        html += '<div class="model-group glass">' +
            '<div class="model-group-header">' +
                '<span class="model-group-title">' + escHtml(pname) + '</span>' +
                '<span class="model-group-count">' + groupModels.length + ' ' + t('models.count') + '</span>' +
            '</div>' +
            '<div class="model-group-list">';

        for (var mi = 0; mi < groupModels.length; mi++) {
            var m = groupModels[mi];
            html += '<div class="model-item">' +
                '<div class="model-info">' +
                    '<span class="model-name">' + escHtml(m.name || m.id) + '</span>' +
                    '<span class="model-id mono">' + escHtml(m.id) + '</span>' +
                '</div>' +
                '<button class="btn btn-secondary btn-sm" onclick="copyText(\'' + jsEsc(m.id) + '\')">' + t('models.copyId') + '</button>' +
            '</div>';
        }

        html += '</div></div>';
    }

    container.innerHTML = html;
}

/* ═══════════════════════════════ Preprocessors ═══════════════════════════════ */

var preprocessorsData = { preprocessors: {}, models: [] };

async function loadPreprocessors() {
    try {
        var data = await api('/admin/preprocessors');
        preprocessorsData = { preprocessors: data.preprocessors || {}, models: data.models || [] };
        renderPreprocessors();
    } catch (e) {
        toast(t('preprocessors.loadFail') + ': ' + e.message, 'error');
    }
}

function renderPreprocessors() {
    var container = document.getElementById('preprocessorsContent');
    var preprocessorIds = Object.keys(preprocessorsData.preprocessors);
    var models = preprocessorsData.models || [];

    var html = '<div class="preprocessors-layout">';

    // Left: Preprocessor config form
    html += '<div class="preprocessor-config-col">';
    html += '<div class="section-sub-header"><h3 data-i18n="preprocessors.configTitle">预处理器配置</h3>';
    html += '<button class="btn btn-primary btn-sm" onclick="showAddPreprocessorModal()">' + t('preprocessors.add') + '</button></div>';

    if (preprocessorIds.length === 0) {
        html += '<div class="empty-state"><div class="empty-icon">&#127918;</div><p>' + t('preprocessors.empty') + '</p></div>';
    } else {
        preprocessorIds.forEach(function(id) {
            var p = preprocessorsData.preprocessors[id];
            html += preprocessorCardHtml(id, p);
        });
    }
    html += '</div>';

    // Right: Model toggle list
    html += '<div class="preprocessor-models-col">';
    html += '<div class="section-sub-header"><h3 data-i18n="preprocessors.modelsTitle">模型开关</h3></div>';

    if (models.length === 0) {
        html += '<div class="empty-state"><div class="empty-icon">&#128269;</div><p>' + t('preprocessors.modelsEmpty') + '</p></div>';
    } else {
        // 按 provider_name 分组
        var grouped = {};
        for (var i = 0; i < models.length; i++) {
            var m = models[i];
            var pname = m.provider_name || m.provider || '-';
            if (!grouped[pname]) grouped[pname] = [];
            grouped[pname].push(m);
        }
        var providerKeys = Object.keys(grouped).sort();
        for (var pi = 0; pi < providerKeys.length; pi++) {
            var pname = providerKeys[pi];
            var groupModels = grouped[pname];
            html += '<div class="model-group glass">' +
                '<div class="model-group-header">' +
                    '<span class="model-group-title">' + escHtml(pname) + '</span>' +
                    '<span class="model-group-count">' + groupModels.length + ' ' + t('models.count') + '</span>' +
                '</div>' +
                '<div class="model-group-list">';
            for (var mi = 0; mi < groupModels.length; mi++) {
                var m = groupModels[mi];
                var checked = m.preprocessor ? 'checked' : '';
                html += '<div class="model-toggle-item">' +
                    '<div class="model-toggle-info">' +
                        '<span class="model-toggle-name">' + escHtml(m.model_id) + '</span>' +
                        '<span class="model-toggle-status ' + (m.preprocessor ? 'on' : 'off') + '">' +
                            (m.preprocessor ? t('preprocessors.modelsOn') : t('preprocessors.modelsOff')) +
                        '</span>' +
                    '</div>' +
                    '<label class="toggle-switch">' +
                        '<input type="checkbox" ' + checked + ' onchange="toggleModelPreprocessor(\'' + jsEsc(m.model_id) + '\', this.checked)">' +
                        '<span class="toggle-slider"></span>' +
                    '</label>' +
                '</div>';
            }
            html += '</div></div>';
        }
    }
    html += '</div>';

    html += '</div>';
    container.innerHTML = html;
}

function preprocessorCardHtml(id, p) {
    return '<div class="preprocessor-card glass">' +
        '<div class="preprocessor-card-header">' +
            '<div class="preprocessor-card-title">' + escHtml(id) + '</div>' +
            '<div class="preprocessor-card-actions">' +
                '<button class="btn btn-secondary btn-sm" onclick="editPreprocessor(\'' + jsEsc(id) + '\')">' + t('common.edit') + '</button>' +
                '<button class="btn btn-danger btn-sm" onclick="deletePreprocessor(\'' + jsEsc(id) + '\')">' + t('preprocessors.delete') + '</button>' +
            '</div>' +
        '</div>' +
        '<div class="preprocessor-card-body">' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.apiBase') + ':</span>' +
                '<code>' + escHtml(p.api_base || '-') + '</code>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.model') + ':</span>' +
                '<code>' + escHtml(p.model || '-') + '</code>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.timeout') + ':</span>' +
                '<span>' + (p.timeout || 30) + 's</span>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.maxImages') + ':</span>' +
                '<span>' + (p.max_images || 1) + '</span>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.maxTokens') + ':</span>' +
                '<span>' + (p.max_tokens || 1024) + '</span>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="status-dot ' + (p.enabled ? 'on' : 'off') + '"></span>' +
                '<span>' + (p.enabled ? t('preprocessors.enabled') : t('preprocessors.disabled')) + '</span>' +
            '</div>' +
        '</div>' +
    '</div>';
}

function preprocessorFormHtml(title, preprocessor, submitAction) {
    preprocessor = preprocessor || {};
    var nameField = preprocessor._id !== undefined
        ? '<div class="form-group"><label>' + t('preprocessors.name') + '</label>' +
            '<input type="text" id="prepName" value="' + escHtml(preprocessor._id) + '" placeholder="' + t('preprocessors.namePlaceholder') + '"></div>'
        : '<div class="form-group"><label>' + t('preprocessors.name') + '</label>' +
            '<input type="text" id="prepName" value="" placeholder="' + t('preprocessors.namePlaceholder') + '"></div>';
    return '<h2>' + title + '</h2>' +
        nameField +
        '<div class="form-group"><label>' + t('preprocessors.apiBase') + '</label>' +
            '<input type="text" id="prepApiBase" value="' + escHtml(preprocessor.api_base || '') + '" placeholder="' + t('preprocessors.apiBasePlaceholder') + '"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.apiKey') + '</label>' +
            '<input type="password" id="prepApiKey" value="' + escHtml(preprocessor.api_key || '') + '" placeholder="' + t('preprocessors.apiKeyPlaceholder') + '"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.model') + '</label>' +
            '<div class="input-row">' +
            '<input type="text" id="prepModel" list="prepModelList" value="' + escHtml(preprocessor.model || '') + '" placeholder="' + t('preprocessors.modelPlaceholder') + '" style="flex:1">' +
            '<datalist id="prepModelList"></datalist>' +
            '<button class="btn btn-secondary btn-sm" type="button" onclick="fetchPreprocessorModels()">' + t('preprocessors.fetchModels') + '</button>' +
            '</div></div>' +
        '<div class="form-group"><label>' + t('preprocessors.timeout') + '</label>' +
            '<input type="number" id="prepTimeout" value="' + (preprocessor.timeout || 30) + '" min="1" max="300"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.maxImages') + '</label>' +
            '<input type="number" id="prepMaxImages" value="' + (preprocessor.max_images || 5) + '" min="1" max="50"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.maxTokens') + '</label>' +
            '<input type="number" id="prepMaxTokens" value="' + (preprocessor.max_tokens || 1024) + '" min="128" max="8192"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.prompt') + '</label>' +
            '<textarea id="prepPrompt" rows="3" style="width:100%;resize:vertical">' + escHtml(preprocessor.prompt || '') + '</textarea></div>' +
        '<div class="form-group"><label><input type="checkbox" id="prepEnabled"' + (preprocessor.enabled === false ? '' : ' checked') + '> ' + t('preprocessors.enabled') + '</label></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('preprocessors.cancel') + '</button>' +
            '<button class="btn btn-primary" id="prepSaveBtn">' + t('preprocessors.save') + '</button></div>';
}

async function fetchPreprocessorModels() {
    var apiBase = document.getElementById('prepApiBase').value.trim();
    var apiKey = document.getElementById('prepApiKey').value.trim();
    if (!apiBase) { toast(t('preprocessors.needApiBase'), 'error'); return; }
    var btn = event.target; btn.disabled = true; btn.textContent = '...';
    try {
        var data = await api('/admin/preprocessors/fetch-models?api_base=' + encodeURIComponent(apiBase) + '&api_key=' + encodeURIComponent(apiKey));
        var models = data.models || [];
        var dl = document.getElementById('prepModelList');
        dl.innerHTML = models.map(function(m) { return '<option value="' + escHtml(m) + '">'; }).join('');
        if (models.length) {
            document.getElementById('prepModel').value = models[0];
            toast(models.length + ' ' + t('preprocessors.modelsFound'), 'success');
        } else {
            toast(t('preprocessors.noModels'), 'warning');
        }
    } catch(e) { toast(t('preprocessors.fetchFail') + ': ' + e.message, 'error'); }
    finally { btn.disabled = false; btn.textContent = t('preprocessors.fetchModels'); }
}

function readPreprocessorForm() {
    return {
        _name: document.getElementById('prepName').value.trim(),
        api_base: document.getElementById('prepApiBase').value.trim(),
        model: document.getElementById('prepModel').value.trim(),
        api_key: document.getElementById('prepApiKey').value.trim(),
        timeout: parseInt(document.getElementById('prepTimeout').value) || 30,
        max_images: parseInt(document.getElementById('prepMaxImages').value) || 5,
        max_tokens: parseInt(document.getElementById('prepMaxTokens').value) || 1024,
        prompt: document.getElementById('prepPrompt').value.trim(),
        enabled: document.getElementById('prepEnabled').checked
    };
}

function showAddPreprocessorModal() {
    document.getElementById('modalContent').innerHTML = preprocessorFormHtml(t('preprocessors.addTitle'), {}, 'addPreprocessor()');
    document.getElementById('prepSaveBtn').onclick = addPreprocessor;
    document.getElementById('modal').style.display = 'flex';
}

function editPreprocessor(id) {
    var p = preprocessorsData.preprocessors[id];
    if (!p) return;
    p._id = id;
    document.getElementById('modalContent').innerHTML = preprocessorFormHtml(t('preprocessors.editTitle'), p, 'updatePreprocessor(\'' + jsEsc(id) + '\')');
    document.getElementById('prepSaveBtn').onclick = function() { updatePreprocessor(id); };
    document.getElementById('modal').style.display = 'flex';
}

async function addPreprocessor() {
    var form = readPreprocessorForm();
    if (!form._name) {
        toast(t('preprocessors.nameRequired'), 'error');
        return;
    }
    if (!form.api_base || !form.model) {
        toast(t('preprocessors.addFail') + ': missing required fields', 'error');
        return;
    }
    var id = form._name;
    delete form._name;
    try {
        await api('/admin/preprocessors/' + encodeURIComponent(id), {
            method: 'PUT',
            body: JSON.stringify(form)
        });
        closeModal();
        loadPreprocessors();
    } catch (e) {
        toast(t('preprocessors.addFail') + ': ' + e.message, 'error');
    }
}

async function updatePreprocessor(id) {
    var form = readPreprocessorForm();
    var newId = form._name;
    if (!newId) {
        toast(t('preprocessors.nameRequired'), 'error');
        return;
    }
    delete form._name;
    try {
        if (newId !== id) {
            // Rename: create new entry, then delete old
            await api('/admin/preprocessors/' + encodeURIComponent(newId), {
                method: 'PUT',
                body: JSON.stringify(form)
            });
            await api('/admin/preprocessors/' + encodeURIComponent(id), {
                method: 'DELETE'
            });
        } else {
            await api('/admin/preprocessors/' + encodeURIComponent(id), {
                method: 'PUT',
                body: JSON.stringify(form)
            });
        }
        closeModal();
        loadPreprocessors();
    } catch (e) {
        toast(t('preprocessors.updateFail') + ': ' + e.message, 'error');
    }
}

async function deletePreprocessor(id) {
    if (!confirm(t('preprocessors.deleteConfirm'))) return;
    try {
        await api('/admin/preprocessors/' + encodeURIComponent(id), { method: 'DELETE' });
        loadPreprocessors();
    } catch (e) {
        toast(t('preprocessors.deleteFail') + ': ' + e.message, 'error');
    }
}

async function toggleModelPreprocessor(modelId, enabled) {
    try {
        await api('/admin/models/preprocessor', {
            method: 'PUT',
            body: JSON.stringify({ model_id: modelId, enabled: enabled })
        });
        loadPreprocessors();
    } catch (e) {
        toast(t('preprocessors.toggleFail') + ': ' + e.message, 'error');
    }
}

/* ═══════════════════════════════ Stats ═══════════════════════════════ */

var _statsTimer = null;
var _statsCharts = [];

function _destroyCharts() {
    for (var i = 0; i < _statsCharts.length; i++) {
        try { _statsCharts[i].destroy(); } catch (e) {}
    }
    _statsCharts = [];
}

async function loadStats() {
    try {
        var stats = await api('/admin/stats');
        renderStats(stats, true);
    } catch (e) {
        toast(t('stats.loadFail') + ': ' + e.message, 'error');
    }
    _startStatsPolling();
}

function _startStatsPolling() {
    if (_statsTimer) clearInterval(_statsTimer);
    _statsTimer = setInterval(async function() {
        try {
            var s = await api('/admin/stats');
            renderStats(s, false);
        } catch (e) {}
    }, 5000);
}

function stopStatsTimer() {
    if (_statsTimer) { clearInterval(_statsTimer); _statsTimer = null; }
    _destroyCharts();
}

async function confirmResetStats() {
    if (!confirm(t('stats.resetConfirm'))) return;
    try {
        await api('/admin/stats/reset', { method: 'POST' });
        await loadStats();
    } catch (e) { toast(t('stats.resetFail') + ': ' + e.message, 'error'); }
}

function renderStats(stats, createCharts) {
    var activeModels = Object.keys(stats.stats_by_model || {}).length;
    var hasData = stats.total_calls > 0;

    // Always update summary cards and toolbar (cheap DOM update)
    var summaryHTML = '<div class="dashboard-toolbar">' +
        '<span>' + t('stats.reset') + ': ' + (stats.last_reset || '-') + '  |  ' + t('stats.autoRefresh') + '</span>' +
        '<button class="btn btn-danger btn-sm" onclick="confirmResetStats()">' + t('stats.resetBtn') + '</button>' +
        '</div>' +
        '<div class="summary-cards">' +
        '<div class="summary-card card-purple"><div class="card-icon">&#9636;</div><div class="card-value">' + stats.total_calls.toLocaleString() + '</div><div class="card-label">' + t('stats.totalCalls') + '</div></div>' +
        '<div class="summary-card card-green"><div class="card-icon">&#10003;</div><div class="card-value">' + stats.success_rate + '%</div><div class="card-label">' + t('stats.successRate') + '</div></div>' +
        '<div class="summary-card card-red"><div class="card-icon">&#9888;</div><div class="card-value">' + stats.failed_calls.toLocaleString() + '</div><div class="card-label">' + t('stats.failedCalls') + '</div></div>' +
        '<div class="summary-card card-blue"><div class="card-icon">&#9881;</div><div class="card-value">' + activeModels + '</div><div class="card-label">' + t('stats.activeModels') + '</div></div>' +
        '</div>';

    // Update activity table
    var log = stats.request_log || [];
    var tableHTML = '<div class="table-card glass"><h3>' + t('stats.realtime') + '</h3>' +
        '<table class="modern-table"><thead><tr>' +
        '<th>' + t('stats.time') + '</th><th>' + t('stats.client') + '</th><th>' + t('stats.key') + '</th><th>' + t('stats.requestedModel') + '</th><th>' + t('stats.model') + '</th><th>' + t('stats.endpoint') + '</th><th>' + t('stats.tokens') + '</th><th>' + t('stats.status') + '</th>' +
        '</tr></thead><tbody>';

    var displayLog = log.slice(0, 40);
    for (var i = 0; i < displayLog.length; i++) {
        var entry = displayLog[i];
        var badge = entry.success ? 'badge-ok' : 'badge-fail';
        var reqModel = entry.requested_model || entry.model;
        var isFallback = reqModel !== entry.model;
        tableHTML += '<tr>' +
            '<td>' + escHtml(entry.time) + '</td>' +
            '<td>' + escHtml(entry.username) + '</td>' +
            '<td class="mono">' + escHtml(entry.api_key) + '</td>' +
            '<td>' + escHtml(reqModel) + '</td>' +
            '<td>' + (isFallback ? '<span style="color:var(--warning)" title="fallback">' + escHtml(entry.model) + '</span>' : escHtml(entry.model)) + '</td>' +
            '<td><span class="badge-endpoint">' + escHtml(entry.endpoint) + '</span></td>' +
            '<td>' + entry.tokens.toLocaleString() + '</td>' +
            '<td><span class="badge ' + badge + '">' + (entry.success ? 'OK' : 'FAIL') + '</span></td>' +
        '</tr>';
    }
    if (log.length === 0) {
        tableHTML += '<tr><td colspan="9" style="text-align:center;color:var(--text-tertiary);padding:24px">' + t('stats.noRecords') + '</td></tr>';
    }
    tableHTML += '</tbody></table></div>';

    if (!hasData) {
        _destroyCharts();
        document.getElementById('statsContent').innerHTML = '<div class="dashboard">' + summaryHTML +
            '<div class="empty-state"><div class="empty-icon">&#128202;</div><p>' + t('stats.noData') + '</p><p class="empty-sub">' + t('stats.noDataHint') + '</p></div>' +
            '</div>';
        return;
    }

    if (createCharts) {
        // Full render: build chart containers and create new Chart instances
        _destroyCharts();
        var dist = stats.distribution || {};
        var tl = stats.timeline || {};

        var html = '<div class="dashboard">' + summaryHTML;
        html += '<div class="charts-row">';
        if (dist.labels && dist.labels.length > 0) {
            html += '<div class="chart-card glass"><h3>' + t('stats.modelDist') + '</h3><div class="chart-wrap"><canvas id="chartPie"></canvas></div></div>';
        }
        if (tl.labels && tl.labels.length > 0) {
            html += '<div class="chart-card glass"><h3>' + t('stats.timeline') + '</h3><div class="chart-wrap"><canvas id="chartLine"></canvas></div></div>';
        }
        html += '</div>' + tableHTML + '</div>';
        document.getElementById('statsContent').innerHTML = html;

        // Create charts
        var chartColors = ['#818cf8','#34d399','#fbbf24','#f472b6','#f87171','#38bdf8','#a78bfa','#fb923c','#6366f1','#4ade80'];
        if (dist.labels && dist.labels.length > 0) {
            var pieCtx = document.getElementById('chartPie');
            if (pieCtx) {
                _statsCharts.push(new Chart(pieCtx, {
                    type: 'doughnut',
                    data: { labels: dist.labels, datasets: [{ data: dist.counts, backgroundColor: chartColors.slice(0, dist.labels.length), borderWidth: 0 }] },
                    options: {
                        responsive: true, maintainAspectRatio: true,
                        plugins: { legend: { position: 'right', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8,
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(), font: { size: 12 } } } }
                    }
                }));
            }
        }
        if (tl.labels && tl.labels.length > 0) {
            var lineCtx = document.getElementById('chartLine');
            if (lineCtx) {
                _statsCharts.push(new Chart(lineCtx, {
                    type: 'bar',
                    data: {
                        labels: tl.labels,
                        datasets: [
                            { label: t('stats.chartSuccess'), data: tl.success, backgroundColor: '#34d399', borderRadius: 4 },
                            { label: t('stats.chartFail'), data: tl.failed, backgroundColor: '#f87171', borderRadius: 4 }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: true,
                        scales: {
                            x: { stacked: true, grid: { display: false }, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 } } },
                            y: { stacked: true, beginAtZero: true, ticks: { stepSize: 1, color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 } } }
                        },
                        plugins: { legend: { labels: { usePointStyle: true, pointStyleWidth: 8,
                            color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(), font: { size: 12 }, padding: 16 } } }
                    }
                }));
            }
        }
    } else {
        // Update mode: only update summary + table via innerHTML, update chart data in-place
        var toolbarEl = document.querySelector('#statsContent .dashboard-toolbar');
        var cardsEl = document.querySelector('#statsContent .summary-cards');
        var tableEl = document.querySelector('#statsContent .table-card');

        var temp = document.createElement('div');
        temp.innerHTML = summaryHTML;
        if (toolbarEl) toolbarEl.replaceWith(temp.querySelector('.dashboard-toolbar'));
        if (cardsEl) cardsEl.replaceWith(temp.querySelector('.summary-cards'));

        if (tableEl) {
            var newTable = document.createElement('div');
            newTable.innerHTML = tableHTML;
            tableEl.replaceWith(newTable.firstElementChild);
        }

        // Update chart data in-place
        var dist = stats.distribution || {};
        var tl = stats.timeline || {};
        for (var c = 0; c < _statsCharts.length; c++) {
            var chart = _statsCharts[c];
            try {
                if (chart.canvas.id === 'chartPie' && dist.labels) {
                    chart.data.labels = dist.labels;
                    chart.data.datasets[0].data = dist.counts;
                    chart.update('none');
                } else if (chart.canvas.id === 'chartLine' && tl.labels) {
                    chart.data.labels = tl.labels;
                    chart.data.datasets[0].data = tl.success;
                    chart.data.datasets[1].data = tl.failed;
                    chart.update('none');
                }
            } catch (e) {}
        }
    }
}

/* ═══════════════════════════════ Model Selector ═══════════════════════════════ */

function modelSelectorHtml(selectedModels, prefix) {
    if (!allModels.length) {
        return '<p style="color:var(--text-tertiary);font-size:13px;">' + t('users.modelsHint') + '</p>';
    }
    var selected = new Set(selectedModels || []);
    var allChecked = selected.has('*');

    var grouped = {};
    for (var i = 0; i < allModels.length; i++) {
        var m = allModels[i];
        var pname = m.provider_name || m.provider;
        if (!grouped[pname]) grouped[pname] = [];
        grouped[pname].push(m);
    }

    var html = '<div class="model-selector">';
    html += '<div class="model-selector-all"><label class="model-selector-item">' +
        '<input type="checkbox" id="' + prefix + '_all"' + (allChecked ? ' checked' : '') + ' onchange="toggleModelAll(\'' + prefix + '\')">' +
        '<strong>' + t('users.allModels') + '</strong> <span style="color:var(--text-tertiary);margin-left:4px;font-size:11px;">(' + t('users.wildcard') + ')</span>' +
    '</label></div>';

    var pnames = Object.keys(grouped);
    for (var gi = 0; gi < pnames.length; gi++) {
        var pname = pnames[gi];
        var pmodels = grouped[pname];
        html += '<div class="model-selector-group"><div class="model-selector-provider">' + escHtml(pname) + '</div>';
        for (var mi = 0; mi < pmodels.length; mi++) {
            var m = pmodels[mi];
            var modelChecked = allChecked || selected.has(m.id);
            html += '<label class="model-selector-item">' +
                '<input type="checkbox" class="' + prefix + '_model" id="' + prefix + '_' + escAttr(m.id) + '" value="' + escAttr(m.id) + '"' + (modelChecked ? ' checked' : '') + (allChecked ? ' disabled' : '') + '>' +
                escHtml(m.name || m.id) +
            '</label>';
        }
        html += '</div>';
    }
    html += '</div>';
    return html;
}

function toggleModelAll(prefix) {
    var allCheckbox = document.getElementById(prefix + '_all');
    var isAll = allCheckbox.checked;
    var modelCheckboxes = document.querySelectorAll('.' + prefix + '_model');
    for (var i = 0; i < modelCheckboxes.length; i++) {
        modelCheckboxes[i].checked = isAll;
        modelCheckboxes[i].disabled = isAll;
    }
}

function readModelSelector(prefix) {
    var allCheckbox = document.getElementById(prefix + '_all');
    if (allCheckbox && allCheckbox.checked) return ['*'];
    var modelCheckboxes = document.querySelectorAll('.' + prefix + '_model');
    var selected = [];
    for (var i = 0; i < modelCheckboxes.length; i++) {
        if (modelCheckboxes[i].checked) selected.push(modelCheckboxes[i].value);
    }
    return selected;
}

/* ═══════════════════════════════ Helpers ═══════════════════════════════ */

function escAttr(str) {
    return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmtModels(models) {
    if (!models || !models.length) return '-';
    if (models.indexOf('*') !== -1) return t('users.allModels');
    return models.map(escHtml).join(', ');
}

function maskKey(key) {
    if (!key || key.length < 12) return key || '';
    return key.slice(0, 8) + '...' + key.slice(-4);
}

async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        toast(t('common.copied'), 'success');
        return;
    }
    // 非安全上下文（HTTP 远程访问）降级方案
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
        document.execCommand('copy');
        toast(t('common.copied'), 'success');
    } catch (e) {
        toast(t('common.copy_failed'), 'error');
    }
    document.body.removeChild(ta);
}

function closeModal() {
    document.getElementById('modal').style.display = 'none';
}

// Close modal on overlay click
document.addEventListener('click', function(e) {
    if (e.target.id === 'modal') closeModal();
});

function escHtml(value) {
    return String(value).replace(/[&<>"']/g, function(ch) {
        return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch];
    });
}

function jsEsc(value) {
    return String(value)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, "\\'")
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029')
        .replace(/"/g, '&quot;');
}

/* ═══════════════════════════════ Init ═══════════════════════════════ */

document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    applyI18n();
    initAuth().catch(function(err) {
        toast(t('auth.initFail') + ': ' + err.message, 'error');
    });
});
