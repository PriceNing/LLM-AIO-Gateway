/* ═══════════════════════════════════════════════════════════════════════
   LLM AIO Gateway Admin — Application
   ═══════════════════════════════════════════════════════════════════════ */

const API_BASE = '';
const SESSION_KEY = 'llm_gateway_admin_session';
const LANG_KEY = 'llm_gateway_lang';
const THEME_KEY = 'llm_gateway_theme';
const SESSION_EXPIRED_EVENT = 'llm-gateway-session-expired';

let authMode = 'login';
let providers = [];
let models = [];
let allModels = [];
let users = [];
let currentLang = localStorage.getItem(LANG_KEY) || 'zh';
let sessionExpiredShown = false;
let serviceVersion = '';
window._requestLogDetails = [];
window._systemLogMeta = null;

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
    'nav.fallbacks': 'Fallback 策略',
    'nav.stats': '统计',
    'nav.preprocessors': '视觉模型注入',
    'nav.imageGeneration': '图像生成',
    'nav.logout': '退出',
    'nav.github': 'GitHub 项目地址',
    'nav.switchLang': '切换语言',
    'nav.switchTheme': '切换主题',
    'nav.changePassword': '修改密码',
    'auth.expired': '登录已过期，请重新登录',

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
    'users.filterModels': '筛选模型或提供商...',
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
    'providers.health': '健康检查',
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
    'providers.requestTimeout': '请求超时（秒）',
    'providers.retryCount': '重试次数',
    'providers.retryBackoff': '重试退避（秒）',
    'providers.extraHeaders': '扩展 Headers (JSON)',
    'providers.forceChatCompletions': '强制使用 Chat Completions（跳过 Responses 探测）',
    'providers.addFail': '新增失败',
    'providers.updateFail': '更新失败',
    'providers.deleteConfirm': '确定要删除这个提供商吗？',
    'providers.deleteFail': '删除失败',
    'providers.refreshOk': '刷新成功，发现 {n} 个模型',
    'providers.refreshFail': '刷新失败',
    'providers.healthOk': '健康检查通过：{n} 个模型，{ms}ms',
    'providers.healthFail': '健康检查失败',
    'providers.refreshAllDone': '刷新完成',
    'providers.refreshAllFail': '刷新失败',

    'models.title': '模型列表',
    'models.empty': '暂无模型',
    'models.search': '搜索模型...',
    'models.refreshAll': '刷新所有模型',
    'models.copyId': '复制 ID',
    'models.test': '测试',
    'models.testRunning': '测试中...',
    'models.testFail': '模型测试失败',
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
    'routing.matchScope': '匹配范围',
    'routing.scopeAny': '全部（兼容旧行为）',
    'routing.scopeUnqualified': '仅裸模型',
    'routing.scopeQualified': '仅限定模型',
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
    'routing.dryRun': 'Dry Run',
    'routing.dryRunTitle': '路由 Dry Run',
    'routing.dryRunUser': '用户名（可选）',
    'routing.dryRunKey': 'API Key 或匹配片段（可选）',
    'routing.dryRunModel': '请求模型',
    'routing.dryRunResolvedModel': '解析后模型（可选）',
    'routing.dryRunSubmit': '运行',
    'routing.dryRunFail': '路由 Dry Run 失败',
    'routing.dryRunNoModel': '请输入请求模型',
    'routing.dryRunMatched': '命中规则',
    'routing.dryRunNoMatch': '未命中规则',
    'routing.dryRunProvider': '目标提供商',
    'routing.dryRunEffective': '最终路由',
    'routing.dryRunReason': '原因',
    'routing.dryRunFallback': 'Fallback 预览',

    'fallbacks.title': 'Fallback 策略',
    'fallbacks.add': '新增策略',
    'fallbacks.empty': '暂无 Fallback 策略',
    'fallbacks.addTitle': '新增 Fallback 策略',
    'fallbacks.editTitle': '编辑 Fallback 策略',
    'fallbacks.name': '策略名称',
    'fallbacks.matchProvider': '匹配提供商',
    'fallbacks.matchModel': '匹配模型',
    'fallbacks.triggers': '触发条件',
    'fallbacks.chain': 'Fallback 链',
    'fallbacks.addTarget': '添加目标',
    'fallbacks.timeout': '超时',
    'fallbacks.attemptTimeout': '尝试超时（秒）',
    'fallbacks.attemptTimeoutHint': '当前上游在出结果前最多等待多久；超时后自动切换到下一目标。默认 60 秒。',
    'fallbacks.connectionError': '连接错误',
    'fallbacks.http429': 'HTTP 429',
    'fallbacks.http5xx': 'HTTP 5xx',
    'fallbacks.http4xx': 'HTTP 4xx',
    'fallbacks.loadFail': '加载 Fallback 策略失败',
    'fallbacks.addFail': '新增 Fallback 策略失败',
    'fallbacks.updateFail': '更新 Fallback 策略失败',
    'fallbacks.deleteFail': '删除 Fallback 策略失败',
    'fallbacks.deleteConfirm': '确定要删除这条 Fallback 策略吗？',
    'fallbacks.save': '保存',
    'fallbacks.cancel': '取消',
    'fallbacks.enabled': '启用',
    'fallbacks.disabled': '禁用',
    'fallbacks.noMatchModel': '请填写匹配模型',
    'fallbacks.delete': '删除',
    'fallbacks.edit': '编辑',

    'stats.title': '调用统计',
    'stats.loadFail': '加载统计失败',
    'stats.totalCalls': '总调用次数',
    'stats.successRate': '硬成功率',
    'stats.healthRate': '健康率',
    'stats.degradedCalls': '降级次数',
    'stats.rejectedCalls': '拒绝次数',
    'stats.cancelledCalls': '中断次数',
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
    'stats.routedModel': '路由目标',
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

    'stats.tabRealtime': '实时监控',
    'stats.tabHistory': '历史记录',
    'stats.historyFrom': '起始日期',
    'stats.historyTo': '结束日期',
    'stats.granularity': '粒度',
    'stats.granHour': '小时',
    'stats.granDay': '天',
    'stats.granWeek': '周',
    'stats.granMonth': '月',
    'stats.query': '查询',
    'stats.periodSummary': '时段汇总',
    'stats.periodCalls': '调用次数',
    'stats.periodTokens': '总 Tokens',
    'stats.periodSuccessRate': '成功率',
    'stats.modelBreakdown': '模型用量明细',
    'stats.userBreakdown': '用户用量明细',
    'stats.historyNoData': '所选时段暂无数据',
    'stats.historyNoDataHint': '尝试调整时间范围或粒度设置',
    'stats.trendChart': '调用趋势',
    'stats.trendCalls': '调用次数',
    'stats.trendTokens': 'Token 数',

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
    'preprocessors.test': '测试',
    'preprocessors.testRunning': '测试中...',
    'preprocessors.testFail': '视觉模型测试失败',
    'imageGeneration.title': '图像生成',
    'imageGeneration.config': '全局生图后端',
    'imageGeneration.models': '模型开关',
    'imageGeneration.save': '保存配置',
    'imageGeneration.test': '测试连接',
    'imageGeneration.testRunning': '测试中...',
    'imageGeneration.testStarted': '正在测试生图连接，部分模型可能需要几分钟',
    'imageGeneration.testFail': '图像生成连接测试失败',
    'imageGeneration.noGenerator': '请先保存生图后端配置',
    'imageGeneration.backendType': '后端类型',
    'imageGeneration.existingModel': '已有提供商模型',
    'imageGeneration.externalModel': '外部模型',
    'imageGeneration.comfyui': 'ComfyUI',
    'imageGeneration.providerModel': '提供商模型',
    'imageGeneration.selectModel': '请选择提供商模型',
    'imageGeneration.apiBase': 'API Base URL',
    'imageGeneration.apiKey': 'API Key',
    'imageGeneration.model': '模型名称',
    'imageGeneration.comfyBase': 'ComfyUI Base URL',
    'imageGeneration.workflow': 'API 格式工作流 JSON',
    'imageGeneration.workflowHint': '支持 ComfyUI 普通工作流和 API Format 工作流；普通工作流会自动转换',
    'imageGeneration.analyzeWorkflow': '解析工作流',
    'imageGeneration.fetchWorkflows': '获取服务器工作流',
    'imageGeneration.fetchingWorkflows': '正在获取...',
    'imageGeneration.savedWorkflow': '服务器工作流',
    'imageGeneration.selectWorkflow': '请选择工作流',
    'imageGeneration.loadWorkflow': '载入所选工作流',
    'imageGeneration.workflowLoaded': '工作流已载入，可以直接解析；普通工作流会自动转换为 API Format',
    'imageGeneration.workflowConverted': '普通 ComfyUI 工作流已自动转换为 API Format',
    'imageGeneration.fetchWorkflowsFail': '获取 ComfyUI 工作流失败',
    'imageGeneration.analyzingWorkflow': '解析中...',
    'imageGeneration.workflowAnalyzed': '已解析工作流节点',
    'imageGeneration.workflowAnalyzeFail': '工作流解析失败',
    'imageGeneration.mapping': '工作流输入映射',
    'imageGeneration.positivePrompt': '正向提示词',
    'imageGeneration.negativePrompt': '负向提示词（可选）',
    'imageGeneration.width': '宽度（可选）',
    'imageGeneration.height': '高度（可选）',
    'imageGeneration.seed': 'Seed（可选）',
    'imageGeneration.steps': '采样步数（可选）',
    'imageGeneration.cfg': 'CFG（可选）',
    'imageGeneration.batchSize': '批量数量（可选）',
    'imageGeneration.outputNode': '图像输出节点',
    'imageGeneration.selectMapping': '请选择节点输入',
    'imageGeneration.autoOutput': '自动查找所有输出节点',
    'imageGeneration.pollInterval': '状态轮询间隔（秒）',
    'imageGeneration.timeout': '超时时间（秒）',
    'imageGeneration.enabled': '启用',
    'imageGeneration.on': '开',
    'imageGeneration.off': '关',
    'imageGeneration.loadFail': '加载图像生成配置失败',
    'imageGeneration.saveFail': '保存图像生成配置失败',
    'imageGeneration.toggleFail': '切换图像生成开关失败',

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
    'models.addCustom': '添加自定义模型',
    'models.addCustomTitle': '添加自定义模型',
    'models.modelId': '模型 ID',
    'models.modelName': '模型名称',
    'models.sourceManual': '手动',
    'models.sourceAuto': '自动',
    'models.delete': '删除模型',
    'models.deleteConfirm': '确定要删除模型 {model} 吗？',
    'models.deleteFail': '删除模型失败',
    'models.makeManual': '转为手动（不会被刷新删除）',
    'models.editModel': '编辑模型',
    'models.saveEdit': '保存',
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
    'nav.fallbacks': 'Fallback Policies',
    'nav.stats': 'Stats',
    'nav.preprocessors': 'Vision Model Injection',
    'nav.imageGeneration': 'Image Generation',
    'nav.logout': 'Logout',
    'nav.github': 'GitHub Project',
    'nav.switchLang': 'Switch Language',
    'nav.switchTheme': 'Toggle Theme',
    'nav.changePassword': 'Change Password',
    'auth.expired': 'Your session has expired. Please log in again.',

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
    'users.filterModels': 'Filter models or providers...',
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
    'providers.health': 'Health',
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
    'providers.requestTimeout': 'Request Timeout (s)',
    'providers.retryCount': 'Retry Count',
    'providers.retryBackoff': 'Retry Backoff (s)',
    'providers.extraHeaders': 'Extra Headers (JSON)',
    'providers.forceChatCompletions': 'Force Chat Completions (skip Responses detection)',
    'providers.addFail': 'Failed to add',
    'providers.updateFail': 'Failed to update',
    'providers.deleteConfirm': 'Delete this provider?',
    'providers.deleteFail': 'Failed to delete',
    'providers.refreshOk': 'Refresh OK — {n} models found',
    'providers.refreshFail': 'Refresh failed',
    'providers.healthOk': 'Health OK: {n} models, {ms}ms',
    'providers.healthFail': 'Health check failed',
    'providers.refreshAllDone': 'Refresh complete',
    'providers.refreshAllFail': 'Refresh failed',

    'models.title': 'Models',
    'models.empty': 'No models found',
    'models.search': 'Search models...',
    'models.refreshAll': 'Refresh All',
    'models.copyId': 'Copy ID',
    'models.test': 'Test',
    'models.testRunning': 'Testing...',
    'models.testFail': 'Model test failed',
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
    'routing.matchScope': 'Match Scope',
    'routing.scopeAny': 'Any (legacy compatible)',
    'routing.scopeUnqualified': 'Unqualified models only',
    'routing.scopeQualified': 'Provider-qualified models only',
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
    'routing.dryRun': 'Dry Run',
    'routing.dryRunTitle': 'Routing Dry Run',
    'routing.dryRunUser': 'Username (optional)',
    'routing.dryRunKey': 'API key or match fragment (optional)',
    'routing.dryRunModel': 'Requested Model',
    'routing.dryRunResolvedModel': 'Resolved Model (optional)',
    'routing.dryRunSubmit': 'Run',
    'routing.dryRunFail': 'Routing dry run failed',
    'routing.dryRunNoModel': 'Requested model is required',
    'routing.dryRunMatched': 'Matched Rule',
    'routing.dryRunNoMatch': 'No Rule Matched',
    'routing.dryRunProvider': 'Target Provider',
    'routing.dryRunEffective': 'Effective Route',
    'routing.dryRunReason': 'Reason',
    'routing.dryRunFallback': 'Fallback Preview',

    'fallbacks.title': 'Fallback Policies',
    'fallbacks.add': 'Add Policy',
    'fallbacks.empty': 'No fallback policies configured',
    'fallbacks.addTitle': 'Add Fallback Policy',
    'fallbacks.editTitle': 'Edit Fallback Policy',
    'fallbacks.name': 'Policy Name',
    'fallbacks.matchProvider': 'Match Provider',
    'fallbacks.matchModel': 'Match Model',
    'fallbacks.triggers': 'Triggers',
    'fallbacks.chain': 'Fallback Chain',
    'fallbacks.addTarget': 'Add Target',
    'fallbacks.timeout': 'Timeout',
    'fallbacks.attemptTimeout': 'Attempt timeout (sec)',
    'fallbacks.attemptTimeoutHint': 'Max wait for the current upstream before switching to the next target. Default 60s.',
    'fallbacks.connectionError': 'Connection Error',
    'fallbacks.http429': 'HTTP 429',
    'fallbacks.http5xx': 'HTTP 5xx',
    'fallbacks.http4xx': 'HTTP 4xx',
    'fallbacks.loadFail': 'Failed to load fallback policies',
    'fallbacks.addFail': 'Failed to add fallback policy',
    'fallbacks.updateFail': 'Failed to update fallback policy',
    'fallbacks.deleteFail': 'Failed to delete fallback policy',
    'fallbacks.deleteConfirm': 'Delete this fallback policy?',
    'fallbacks.save': 'Save',
    'fallbacks.cancel': 'Cancel',
    'fallbacks.enabled': 'Enabled',
    'fallbacks.disabled': 'Disabled',
    'fallbacks.noMatchModel': 'Match model is required',
    'fallbacks.delete': 'Delete',
    'fallbacks.edit': 'Edit',

    'stats.title': 'Statistics',
    'stats.loadFail': 'Failed to load stats',
    'stats.totalCalls': 'Total Calls',
    'stats.successRate': 'Hard Success Rate',
    'stats.healthRate': 'Health Rate',
    'stats.degradedCalls': 'Degraded Calls',
    'stats.rejectedCalls': 'Rejected Calls',
    'stats.cancelledCalls': 'Cancelled Calls',
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
    'stats.routedModel': 'Routed Target',
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

    'stats.tabRealtime': 'Realtime',
    'stats.tabHistory': 'History',
    'stats.historyFrom': 'From',
    'stats.historyTo': 'To',
    'stats.granularity': 'Granularity',
    'stats.granHour': 'Hour',
    'stats.granDay': 'Day',
    'stats.granWeek': 'Week',
    'stats.granMonth': 'Month',
    'stats.query': 'Query',
    'stats.periodSummary': 'Period Summary',
    'stats.periodCalls': 'Total Calls',
    'stats.periodTokens': 'Total Tokens',
    'stats.periodSuccessRate': 'Success Rate',
    'stats.modelBreakdown': 'Model Breakdown',
    'stats.userBreakdown': 'User Breakdown',
    'stats.historyNoData': 'No data for selected period',
    'stats.historyNoDataHint': 'Try adjusting the date range or granularity',
    'stats.loading': 'Loading...',
    'stats.trendChart': 'Call Trend',
    'stats.trendCalls': 'Calls',
    'stats.trendTokens': 'Tokens',

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
    'preprocessors.test': 'Test',
    'preprocessors.testRunning': 'Testing...',
    'preprocessors.testFail': 'Vision model test failed',
    'imageGeneration.title': 'Image Generation',
    'imageGeneration.config': 'Global Image Backend',
    'imageGeneration.models': 'Model Toggles',
    'imageGeneration.save': 'Save Configuration',
    'imageGeneration.test': 'Test Connection',
    'imageGeneration.testRunning': 'Testing...',
    'imageGeneration.testStarted': 'Testing the image backend; some models may take several minutes',
    'imageGeneration.testFail': 'Image generation connection test failed',
    'imageGeneration.noGenerator': 'Save the image backend configuration first',
    'imageGeneration.backendType': 'Backend Type',
    'imageGeneration.existingModel': 'Existing Provider Model',
    'imageGeneration.externalModel': 'External Model',
    'imageGeneration.comfyui': 'ComfyUI',
    'imageGeneration.providerModel': 'Provider Model',
    'imageGeneration.selectModel': 'Select a provider model',
    'imageGeneration.apiBase': 'API Base URL',
    'imageGeneration.apiKey': 'API Key',
    'imageGeneration.model': 'Model Name',
    'imageGeneration.comfyBase': 'ComfyUI Base URL',
    'imageGeneration.workflow': 'API-format Workflow JSON',
    'imageGeneration.workflowHint': 'Both regular ComfyUI workflows and API-format workflows are supported; regular workflows are converted automatically',
    'imageGeneration.analyzeWorkflow': 'Analyze Workflow',
    'imageGeneration.fetchWorkflows': 'Fetch Server Workflows',
    'imageGeneration.fetchingWorkflows': 'Fetching...',
    'imageGeneration.savedWorkflow': 'Server Workflow',
    'imageGeneration.selectWorkflow': 'Select a workflow',
    'imageGeneration.loadWorkflow': 'Load Selected Workflow',
    'imageGeneration.workflowLoaded': 'Workflow loaded and ready to analyze; regular workflows are converted to API format automatically',
    'imageGeneration.workflowConverted': 'Regular ComfyUI workflow converted to API format automatically',
    'imageGeneration.fetchWorkflowsFail': 'Failed to fetch ComfyUI workflows',
    'imageGeneration.analyzingWorkflow': 'Analyzing...',
    'imageGeneration.workflowAnalyzed': 'Workflow nodes analyzed',
    'imageGeneration.workflowAnalyzeFail': 'Workflow analysis failed',
    'imageGeneration.mapping': 'Workflow Input Mapping',
    'imageGeneration.positivePrompt': 'Positive Prompt',
    'imageGeneration.negativePrompt': 'Negative Prompt (optional)',
    'imageGeneration.width': 'Width (optional)',
    'imageGeneration.height': 'Height (optional)',
    'imageGeneration.seed': 'Seed (optional)',
    'imageGeneration.steps': 'Sampling Steps (optional)',
    'imageGeneration.cfg': 'CFG (optional)',
    'imageGeneration.batchSize': 'Batch Size (optional)',
    'imageGeneration.outputNode': 'Image Output Node',
    'imageGeneration.selectMapping': 'Select a node input',
    'imageGeneration.autoOutput': 'Discover all output nodes automatically',
    'imageGeneration.pollInterval': 'Status Poll Interval (seconds)',
    'imageGeneration.timeout': 'Timeout (seconds)',
    'imageGeneration.enabled': 'Enabled',
    'imageGeneration.on': 'ON',
    'imageGeneration.off': 'OFF',
    'imageGeneration.loadFail': 'Failed to load image generation settings',
    'imageGeneration.saveFail': 'Failed to save image generation settings',
    'imageGeneration.toggleFail': 'Failed to toggle image generation',

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
    'models.addCustom': 'Add Custom Model',
    'models.addCustomTitle': 'Add Custom Model',
    'models.modelId': 'Model ID',
    'models.modelName': 'Model Name',
    'models.sourceManual': 'Manual',
    'models.sourceAuto': 'Auto',
    'models.delete': 'Delete Model',
    'models.deleteConfirm': 'Delete model {model}?',
    'models.deleteFail': 'Failed to delete model',
    'models.makeManual': 'Make manual (survives refresh)',
    'models.editModel': 'Edit Model',
    'models.saveEdit': 'Save',
}
};

Object.assign(I18N.zh, {
    'stats.details': '详情',
    'stats.requestDetails': '请求详情',
    'stats.statusOk': 'OK',
    'stats.statusFail': 'FAIL',
    'stats.statusPartial': 'PARTIAL',
    'stats.statusDegraded': 'DEGRADED',
    'stats.statusRejected': 'REJECTED',
    'stats.statusCancelled': 'CANCELLED',
    'stats.basicInfo': '基础信息',
    'stats.routingInfo': '路由 / Fallback',
    'stats.errorInfo': '错误信息',
    'stats.fullTime': '完整时间',
    'stats.provider': '提供商',
    'stats.stream': '流式请求',
    'stats.partialOutput': '已输出部分内容',
    'stats.fallbackStatus': 'Fallback 状态',
    'stats.fallbackReason': 'Fallback 原因',
    'stats.fallbackAttempts': 'Fallback 链路',
    'stats.fallbackAttempt': '节点',
    'stats.fallbackAttemptStarted': '尝试中',
    'stats.fallbackAttemptSuccess': '成功',
    'stats.fallbackAttemptFailed': '失败',
    'stats.responsesStateful': 'Responses 状态会话',
    'stats.responsesStateMarkers': '状态标记',
    'stats.fallbackSafetyDecision': 'Fallback 安全决策',
    'stats.statefulFallbackBlocked': '已阻止跨提供商 Fallback',
    'stats.statefulFallbackBlockedCalls': '状态会话 Fallback 阻止次数',
    'stats.routingMatched': '命中路由',
    'stats.routingRule': '路由规则',
    'stats.routingReason': '路由原因',
    'stats.responsesMode': 'Responses 模式',
    'stats.nativeAttempted': '已尝试原生 Responses',
    'stats.nativeFailureEndpoint': '原生失败端点',
    'stats.nativeFailureStatus': '原生失败状态码',
    'stats.nativeFailureReason': '原生失败原因',
    'stats.nativeFailureMessage': '原生失败消息',
    'stats.errorTrigger': '错误触发类型',
    'stats.errorStage': '错误阶段',
    'stats.errorMessage': '错误消息',
    'stats.attemptedModel': '尝试模型',
    'stats.attemptedProvider': '尝试提供商',
    'testResult.title': '测试结果',
    'testResult.status': '状态',
    'testResult.latency': '延迟',
    'testResult.model': '模型',
    'testResult.provider': '提供商',
    'testResult.preview': '响应预览',
    'testResult.error': '错误',
    'testResult.usage': 'Token 用量',
    'testResult.ok': '可用',
    'testResult.fail': '失败'
});

Object.assign(I18N.en, {
    'stats.details': 'Details',
    'stats.requestDetails': 'Request Details',
    'stats.statusOk': 'OK',
    'stats.statusFail': 'FAIL',
    'stats.statusPartial': 'PARTIAL',
    'stats.statusDegraded': 'DEGRADED',
    'stats.statusRejected': 'REJECTED',
    'stats.statusCancelled': 'CANCELLED',
    'stats.basicInfo': 'Basic',
    'stats.routingInfo': 'Routing / Fallback',
    'stats.errorInfo': 'Error',
    'stats.fullTime': 'Full Time',
    'stats.provider': 'Provider',
    'stats.stream': 'Stream',
    'stats.partialOutput': 'Partial Output',
    'stats.routingMatched': 'Routing Matched',
    'stats.routingRule': 'Routing Rule',
    'stats.routingReason': 'Routing Reason',
    'stats.responsesMode': 'Responses Mode',
    'stats.nativeAttempted': 'Native Responses Attempted',
    'stats.nativeFailureEndpoint': 'Native Failure Endpoint',
    'stats.nativeFailureStatus': 'Native Failure Status',
    'stats.nativeFailureReason': 'Native Failure Reason',
    'stats.nativeFailureMessage': 'Native Failure Message',
    'stats.fallbackStatus': 'Fallback Status',
    'stats.fallbackReason': 'Fallback Reason',
    'stats.fallbackAttempts': 'Fallback Chain',
    'stats.fallbackAttempt': 'Attempt',
    'stats.fallbackAttemptStarted': 'Started',
    'stats.fallbackAttemptSuccess': 'Success',
    'stats.fallbackAttemptFailed': 'Failed',
    'stats.responsesStateful': 'Stateful Responses Session',
    'stats.responsesStateMarkers': 'State Markers',
    'stats.fallbackSafetyDecision': 'Fallback Safety Decision',
    'stats.statefulFallbackBlocked': 'Cross-provider fallback blocked',
    'stats.statefulFallbackBlockedCalls': 'Stateful fallback blocks',
    'stats.errorTrigger': 'Error Trigger',
    'stats.errorStage': 'Error Stage',
    'stats.errorMessage': 'Error Message',
    'stats.attemptedModel': 'Attempted Model',
    'stats.attemptedProvider': 'Attempted Provider',
    'testResult.title': 'Test Result',
    'testResult.status': 'Status',
    'testResult.latency': 'Latency',
    'testResult.model': 'Model',
    'testResult.provider': 'Provider',
    'testResult.preview': 'Preview',
    'testResult.error': 'Error',
    'testResult.usage': 'Usage',
    'testResult.ok': 'OK',
    'testResult.fail': 'Failed'
});

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
    if (res.status === 401 && isSessionAuthError(data.detail)) handleSessionExpired();
    if (!res.ok) throw new Error(data.detail || 'HTTP ' + res.status);
    return data;
}

function isSessionAuthError(detail) {
    return detail === 'Missing session token'
        || detail === 'Invalid or expired session'
        || detail === 'Admin disabled';
}

function handleSessionExpired() {
    if (!getToken()) return;
    localStorage.removeItem(SESSION_KEY);
    if (sessionExpiredShown) return;
    sessionExpiredShown = true;
    window.dispatchEvent(new CustomEvent(SESSION_EXPIRED_EVENT));
}

function showAuthView(message) {
    stopStatsTimer();
    closeModal();
    document.getElementById('appView').style.display = 'none';
    document.getElementById('authView').style.display = 'flex';
    document.getElementById('adminPassword').value = '';
    document.getElementById('adminPassword').focus();
    updateAuthHint();
    if (message) toast(message, 'error');
}

function updateAuthHint() {
    document.getElementById('authHint').textContent = authMode === 'setup'
        ? t('auth.hintSetup')
        : t('auth.hint');
}

async function initAuth() {
    try {
        var status = await api('/auth/status');
        serviceVersion = status.version || '';
        updateServiceVersion();
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
    sessionExpiredShown = false;
    document.getElementById('authView').style.display = 'none';
    document.getElementById('appView').style.display = 'block';
    document.getElementById('currentAdmin').textContent = admin.display_name || admin.username;
    updateServiceVersion();
    loadAll();
}

function updateServiceVersion() {
    var el = document.getElementById('serviceVersion');
    if (!el) return;
    el.textContent = serviceVersion ? ('v' + serviceVersion) : '';
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
    if (section === 'fallbacks') { loadFallbackPolicies(); loadModels(); loadProviders(); }
    if (section === 'stats') loadStats();
    if (section === 'preprocessors') loadPreprocessors();
    if (section === 'image-generation') loadImageGeneration();
    if (section === 'request-logs') loadRequestLogs();
    if (section === 'system-logs') loadSystemLogMeta();
    if (section === 'config') {/* lazy load */}
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
                        '<div class="key-main"><div class="key-models"><strong>' + escHtml(key.name) + '</strong> ' + fmtModels(key.allowed_models) + '</div>' +
                        '<code>' + escHtml(maskKey(key.key)) + '</code></div>' +
                        '<div class="key-actions"><span class="key-stats">' + t('users.calls') + ' ' + ((key.stats && key.stats.total_calls) || 0).toLocaleString() + '</span>' +
                        '<button class="btn btn-secondary btn-xs" data-editkey="' + encodeURIComponent(JSON.stringify({u: user.username, k: key.key, n: key.name, m: key.allowed_models || []})) + '" onclick="editUserKeyFromBtn(this)">' + t('users.keyEdit') + '</button>' +
                        '<button class="btn btn-secondary btn-xs" onclick="copyText(\'' + jsEsc(key.key) + '\')">' + t('common.copy') + '</button>' +
                        '<button class="btn btn-danger btn-xs" onclick="deleteUserKey(\'' + jsEsc(user.username) + '\',\'' + jsEsc(key.key) + '\')">' + t('common.delete') + '</button></div>' +
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
                        '<span class="label" style="margin-left:8px">' + t('routing.matchScope') + ':</span><code>' + escHtml(routingScopeLabel(rule.match_scope)) + '</code>' +
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

function routingScopeLabel(scope) {
    return t(scope === 'unqualified' ? 'routing.scopeUnqualified' : scope === 'qualified' ? 'routing.scopeQualified' : 'routing.scopeAny');
}

function modelsForProvider(providerId) {
    var seen = {};
    return (allModels || []).filter(function(m) {
        return !providerId || m.provider === providerId;
    }).map(function(m) {
        return modelValueForProvider(m, providerId || '');
    }).filter(function(id) {
        if (seen[id]) return false;
        seen[id] = true;
        return true;
    });
}

function modelValueForProvider(model, providerId) {
    var id = model.id || '';
    if (providerId && model.provider === providerId) {
        return stripProviderPrefix(id, providerId);
    }
    return id;
}

function stripProviderPrefix(modelId, providerId) {
    var prefix = providerId ? providerId + '/' : '';
    return prefix && modelId.indexOf(prefix) === 0 ? modelId.slice(prefix.length) : modelId;
}

function knownModelIds() {
    var ids = [];
    for (var i = 0; i < (allModels || []).length; i++) {
        var m = allModels[i];
        ids.push(m.id);
        if (m.provider) ids.push(stripProviderPrefix(m.id, m.provider));
    }
    return ids;
}

function isKnownModelId(modelId) {
    return knownModelIds().indexOf(modelId) !== -1;
}

function modelSelectHtml(id, selected, includeWildcard, providerId) {
    var models = modelsForProvider(providerId || '');
    var opts = includeWildcard ? '<option value="*">* (' + t('routing.matchModelHint') + ')</option>' : '<option value="">--</option>';
    var found = selected === '*' || selected === '';
    for (var i = 0; i < models.length; i++) {
        var sel = models[i] === selected ? ' selected' : '';
        if (models[i] === selected) found = true;
        opts += '<option value="' + escHtml(models[i]) + '"' + sel + '>' + escHtml(models[i]) + '</option>';
    }
    // Preserve custom values (e.g. wildcards) when editing
    if (selected && !found && !providerId) {
        opts += '<option value="' + escHtml(selected) + '" selected>' + escHtml(selected) + '</option>';
    }
    return '<select id="' + id + '" style="width:100%">' + opts + '</select>';
}

function providerSelectHtml(id, selected, onchange) {
    var provs = (providers || []).map(function(p) { return p.id; });
    var opts = '<option value="">' + t('routing.targetProvider') + '</option>';
    for (var i = 0; i < provs.length; i++) {
        var sel = provs[i] === selected ? ' selected' : '';
        opts += '<option value="' + escHtml(provs[i]) + '"' + sel + '>' + escHtml(provs[i]) + '</option>';
    }
    var handler = onchange ? ' onchange="' + onchange + '"' : '';
    return '<select id="' + id + '" style="width:100%"' + handler + '>' + opts + '</select>';
}

function refreshModelSelectForProvider(modelSelectId, providerSelectId) {
    var modelSelect = document.getElementById(modelSelectId);
    var providerSelect = document.getElementById(providerSelectId);
    if (!modelSelect || !providerSelect) return;
    var providerId = providerSelect.value.trim();
    var current = stripProviderPrefix(modelSelect.value, providerId);
    var replacement = document.createElement('div');
    replacement.innerHTML = modelSelectHtml(modelSelectId, current, false, providerId);
    modelSelect.replaceWith(replacement.firstChild);
}

function populateModelDatalistForProvider(datalistId, providerId, includeWildcard) {
    var dl = document.getElementById(datalistId);
    if (!dl) return;
    var modelIds = modelsForProvider(providerId || '');
    var opts = includeWildcard ? '<option value="*">' : '';
    for (var i = 0; i < modelIds.length; i++) {
        opts += '<option value="' + escHtml(modelIds[i]) + '">';
    }
    dl.innerHTML = opts;
}

function clearKnownModelOutsideProvider(inputId, providerId) {
    var input = document.getElementById(inputId);
    if (!input || !providerId) return;
    var selectedModel = stripProviderPrefix(input.value.trim(), providerId);
    if (selectedModel !== input.value.trim()) input.value = selectedModel;
    if (!selectedModel || selectedModel === '*' || !isKnownModelId(selectedModel)) return;
    if (modelsForProvider(providerId).indexOf(selectedModel) === -1) {
        input.value = '';
    }
}

function refreshModelDatalistForProvider(inputId, datalistId, providerSelectId, includeWildcard) {
    var providerSelect = document.getElementById(providerSelectId);
    var providerId = providerSelect ? providerSelect.value.trim() : '';
    clearKnownModelOutsideProvider(inputId, providerId);
    populateModelDatalistForProvider(datalistId, providerId, includeWildcard);
}

function refreshFallbackMatchModelsForProvider() {
    refreshModelDatalistForProvider('fallbackMatchModel', 'fallbackModelList', 'fallbackMatchProvider', true);
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
            '<input type="text" id="ruleMatchModel" list="matchModelList" value="' + escHtml(rule.match_model || '') + '" placeholder="*" autocomplete="off">' +
            '<datalist id="matchModelList"></datalist></div>' +
        '<div class="form-group"><label>' + t('routing.matchScope') + '</label>' +
            '<select id="ruleMatchScope" style="width:100%">' +
                '<option value="any"' + ((rule.match_scope || 'any') === 'any' ? ' selected' : '') + '>' + t('routing.scopeAny') + '</option>' +
                '<option value="unqualified"' + (rule.match_scope === 'unqualified' ? ' selected' : '') + '>' + t('routing.scopeUnqualified') + '</option>' +
                '<option value="qualified"' + (rule.match_scope === 'qualified' ? ' selected' : '') + '>' + t('routing.scopeQualified') + '</option>' +
            '</select></div>' +
        '<div class="form-group"><label>' + t('routing.targetProvider') + '</label>' +
            providerSelectHtml('ruleTargetProvider', rule.target_provider || '', "refreshModelSelectForProvider('ruleTargetModel','ruleTargetProvider')") + '</div>' +
        '<div class="form-group"><label>' + t('routing.targetModel') + '</label>' +
            modelSelectHtml('ruleTargetModel', rule.target_model || '', false, rule.target_provider || '') + '</div>' +
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
        match_scope: document.getElementById('ruleMatchScope').value,
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
    populateModelDatalistForProvider('matchModelList', '', true);
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

var fallbackPolicies = [];

async function loadFallbackPolicies() {
    try {
        var data = await api('/admin/fallback-policies');
        fallbackPolicies = data.policies || [];
        renderFallbackPolicies();
    } catch (e) {
        toast(t('fallbacks.loadFail') + ': ' + e.message, 'error');
    }
}

function renderFallbackPolicies() {
    var container = document.getElementById('fallbackPoliciesList');
    if (!container) return;
    if (!fallbackPolicies.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#128737;</div><p>' + t('fallbacks.empty') + '</p></div>';
        return;
    }
    container.innerHTML = fallbackPolicies.map(function(policy) {
        var triggers = policy.triggers || {};
        var enabledTriggers = Object.keys(triggers).filter(function(key) { return triggers[key]; });
        var chain = policy.chain || [];
        return '<div class="routing-card glass">' +
            '<div class="card-top">' +
                '<div>' +
                    '<div class="card-title">' + escHtml(policy.name || policy.id || '') + '</div>' +
                    '<div class="card-meta" style="margin-top:4px;">' +
                        '<span class="status-dot ' + (policy.enabled ? 'on' : 'off') + '"></span>' +
                        '<span>' + (policy.enabled ? t('fallbacks.enabled') : t('fallbacks.disabled')) + '</span>' +
                        '<span>' + escHtml(enabledTriggers.join(', ') || '-') + '</span>' +
                        '<span>' + t('fallbacks.attemptTimeout') + ': ' + escHtml(String(policy.attempt_timeout != null ? policy.attempt_timeout : 60)) + 's</span>' +
                    '</div>' +
                '</div>' +
                '<div class="card-actions">' +
                    '<button class="btn btn-secondary btn-sm" onclick="showEditFallbackPolicyModal(\'' + jsEsc(policy.id) + '\')">' + t('fallbacks.edit') + '</button>' +
                    '<button class="btn btn-danger btn-sm" onclick="deleteFallbackPolicy(\'' + jsEsc(policy.id) + '\')">' + t('fallbacks.delete') + '</button>' +
                '</div>' +
            '</div>' +
            '<div class="routing-rule-detail">' +
                '<div class="routing-arrow">' +
                    '<div class="routing-from"><span class="label">' + t('fallbacks.matchModel') + ':</span><code>' + escHtml(policy.match_model || '*') + '</code>' +
                    (policy.match_provider ? ' @ <code>' + escHtml(policy.match_provider) + '</code>' : '') + '</div>' +
                    '<span class="arrow">&#8594;</span>' +
                    '<div class="routing-to"><span class="label">' + t('fallbacks.chain') + ':</span> ' + fallbackChainText(chain) + '</div>' +
                '</div>' +
            '</div>' +
        '</div>';
    }).join('');
}

function fallbackChainText(chain) {
    if (!chain || !chain.length) return '<code>-</code>';
    return chain.map(function(target) {
        if (typeof target === 'string') return '<code>' + escHtml(target) + '</code>';
        return '<code>' + escHtml((target.provider_id ? target.provider_id + '/' : '') + (target.model || '')) + '</code>';
    }).join(' <span class="arrow">&#8594;</span> ');
}

function fallbackPolicyFormHtml(title, policy) {
    policy = policy || {};
    var triggers = Object.assign({ timeout: true, connection_error: true, http_429: true, http_5xx: true, http_4xx: false }, policy.triggers || {});
    var chain = policy.chain && policy.chain.length ? policy.chain : [{ model: '', provider_id: '' }];
    return '<h2>' + title + '</h2>' +
        '<div class="form-group"><label>' + t('fallbacks.name') + '</label>' +
            '<input type="text" id="fallbackName" value="' + escHtml(policy.name || '') + '"></div>' +
        '<div class="form-group"><label><input type="checkbox" id="fallbackEnabled"' + (policy.enabled === false ? '' : ' checked') + '> ' + t('fallbacks.enabled') + '</label></div>' +
        '<div class="form-group"><label>' + t('fallbacks.matchProvider') + '</label>' +
            providerSelectHtml('fallbackMatchProvider', policy.match_provider || '', 'refreshFallbackMatchModelsForProvider()') + '</div>' +
        '<div class="form-group"><label>' + t('fallbacks.matchModel') + '</label>' +
            '<input type="text" id="fallbackMatchModel" list="fallbackModelList" value="' + escHtml(policy.match_model || '*') + '" placeholder="*" autocomplete="off">' +
            '<datalist id="fallbackModelList"></datalist></div>' +
        '<div class="form-group"><label>' + t('fallbacks.attemptTimeout') + '</label>' +
            '<input type="number" id="fallbackAttemptTimeout" min="5" max="3600" step="1" value="' +
                escHtml(String(policy.attempt_timeout != null ? policy.attempt_timeout : 60)) + '">' +
            '<div class="form-hint muted" style="margin-top:6px;font-size:12px;">' + t('fallbacks.attemptTimeoutHint') + '</div></div>' +
        '<div class="form-group"><label>' + t('fallbacks.triggers') + '</label>' +
            '<div class="checkbox-grid">' +
                fallbackTriggerCheckbox('timeout', t('fallbacks.timeout'), triggers.timeout) +
                fallbackTriggerCheckbox('connection_error', t('fallbacks.connectionError'), triggers.connection_error) +
                fallbackTriggerCheckbox('http_429', t('fallbacks.http429'), triggers.http_429) +
                fallbackTriggerCheckbox('http_5xx', t('fallbacks.http5xx'), triggers.http_5xx) +
                fallbackTriggerCheckbox('http_4xx', t('fallbacks.http4xx'), triggers.http_4xx) +
            '</div></div>' +
        '<div class="form-group"><label>' + t('fallbacks.chain') + '</label>' +
            '<div id="fallbackChainRows">' + chain.map(function(target, index) { return fallbackChainRowHtml(target, index); }).join('') + '</div>' +
            '<button class="btn btn-secondary btn-sm" type="button" onclick="addFallbackTargetRow()">' + t('fallbacks.addTarget') + '</button></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('fallbacks.cancel') + '</button>' +
            '<button class="btn btn-primary" id="fallbackSaveBtn">' + t('fallbacks.save') + '</button></div>';
}

function fallbackTriggerCheckbox(id, label, checked) {
    return '<label><input type="checkbox" class="fallback-trigger" value="' + id + '"' + (checked ? ' checked' : '') + '> ' + label + '</label>';
}

function fallbackChainRowHtml(target, index) {
    target = target || {};
    if (typeof target === 'string') target = { model: target, provider_id: '' };
    var providerId = target.provider_id || target.target_provider || '';
    return '<div class="fallback-chain-row" data-index="' + index + '">' +
        providerSelectHtml('fallbackProvider' + index, providerId, "refreshModelSelectForProvider('fallbackModel" + index + "','fallbackProvider" + index + "')") +
        modelSelectHtml('fallbackModel' + index, target.model || target.target_model || '', false, providerId) +
        '<button class="btn btn-secondary btn-sm" type="button" onclick="moveFallbackTargetRow(this, -1)">&#8593;</button>' +
        '<button class="btn btn-secondary btn-sm" type="button" onclick="moveFallbackTargetRow(this, 1)">&#8595;</button>' +
        '<button class="btn btn-danger btn-sm" type="button" onclick="removeFallbackTargetRow(this)">' + t('common.delete') + '</button>' +
    '</div>';
}

function populateFallbackModelDatalist(providerId) {
    populateModelDatalistForProvider('fallbackModelList', providerId || '', true);
}

function addFallbackTargetRow() {
    var rows = document.getElementById('fallbackChainRows');
    var index = rows.querySelectorAll('.fallback-chain-row').length;
    rows.insertAdjacentHTML('beforeend', fallbackChainRowHtml({}, index));
}

function removeFallbackTargetRow(btn) {
    var rows = document.getElementById('fallbackChainRows');
    if (rows.querySelectorAll('.fallback-chain-row').length <= 1) return;
    btn.closest('.fallback-chain-row').remove();
}

function moveFallbackTargetRow(btn, direction) {
    var row = btn.closest('.fallback-chain-row');
    if (direction < 0 && row.previousElementSibling) row.parentNode.insertBefore(row, row.previousElementSibling);
    if (direction > 0 && row.nextElementSibling) row.parentNode.insertBefore(row.nextElementSibling, row);
}

function readFallbackPolicyForm() {
    var triggers = {};
    document.querySelectorAll('.fallback-trigger').forEach(function(input) { triggers[input.value] = input.checked; });
    var chain = [];
    document.querySelectorAll('.fallback-chain-row').forEach(function(row) {
        var provider = row.querySelector('select[id^="fallbackProvider"]').value.trim();
        var model = row.querySelector('select[id^="fallbackModel"]').value.trim();
        if (model) chain.push({ provider_id: provider, model: model });
    });
    var attemptTimeoutRaw = document.getElementById('fallbackAttemptTimeout').value;
    var attemptTimeout = parseInt(attemptTimeoutRaw, 10);
    if (!isFinite(attemptTimeout) || attemptTimeout <= 0) attemptTimeout = 60;
    if (attemptTimeout < 5) attemptTimeout = 5;
    if (attemptTimeout > 3600) attemptTimeout = 3600;
    return {
        name: document.getElementById('fallbackName').value.trim() || 'New Fallback Policy',
        enabled: document.getElementById('fallbackEnabled').checked,
        match_provider: document.getElementById('fallbackMatchProvider').value.trim(),
        match_model: document.getElementById('fallbackMatchModel').value.trim() || '*',
        attempt_timeout: attemptTimeout,
        triggers: triggers,
        chain: chain
    };
}

function showAddFallbackPolicyModal() {
    document.getElementById('modalContent').innerHTML = fallbackPolicyFormHtml(t('fallbacks.addTitle'), {});
    refreshFallbackMatchModelsForProvider();
    document.getElementById('fallbackSaveBtn').onclick = addFallbackPolicy;
    document.getElementById('modal').style.display = 'flex';
}

function showEditFallbackPolicyModal(policyId) {
    var policy = fallbackPolicies.find(function(p) { return p.id === policyId; });
    if (!policy) return;
    document.getElementById('modalContent').innerHTML = fallbackPolicyFormHtml(t('fallbacks.editTitle'), policy);
    refreshFallbackMatchModelsForProvider();
    document.getElementById('fallbackSaveBtn').onclick = function() { updateFallbackPolicy(policyId); };
    document.getElementById('modal').style.display = 'flex';
}

async function addFallbackPolicy() {
    var form = readFallbackPolicyForm();
    if (!form.match_model) { toast(t('fallbacks.noMatchModel'), 'error'); return; }
    try {
        await api('/admin/fallback-policies', { method: 'POST', body: JSON.stringify(form) });
        closeModal();
        loadFallbackPolicies();
    } catch (e) { toast(t('fallbacks.addFail') + ': ' + e.message, 'error'); }
}

async function updateFallbackPolicy(policyId) {
    var form = readFallbackPolicyForm();
    if (!form.match_model) { toast(t('fallbacks.noMatchModel'), 'error'); return; }
    try {
        await api('/admin/fallback-policies/' + encodeURIComponent(policyId), { method: 'PUT', body: JSON.stringify(form) });
        closeModal();
        loadFallbackPolicies();
    } catch (e) { toast(t('fallbacks.updateFail') + ': ' + e.message, 'error'); }
}

async function deleteFallbackPolicy(policyId) {
    if (!confirm(t('fallbacks.deleteConfirm'))) return;
    try {
        await api('/admin/fallback-policies/' + encodeURIComponent(policyId), { method: 'DELETE' });
        loadFallbackPolicies();
    } catch (e) { toast(t('fallbacks.deleteFail') + ': ' + e.message, 'error'); }
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
                        '<span>' + escHtml((p.request_timeout || 120) + 's / retry ' + (p.retry_count || 0)) + '</span>' +
                    '</div>' +
                '</div>' +
                '<span class="provider-type">' + escHtml(p.provider_type) + '</span>' +
            '</div>' +
            '<div class="card-actions">' +
                '<button class="btn btn-secondary btn-sm" onclick="editProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.edit') + '</button>' +
                '<button class="btn btn-secondary btn-sm" onclick="addCustomModel(\'' + jsEsc(p.id) + '\')">' + t('models.addCustom') + '</button>' +
                '<button class="btn btn-secondary btn-sm" onclick="refreshProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.refresh') + '</button>' +
                '<button class="btn btn-secondary btn-sm" onclick="checkProviderHealth(\'' + jsEsc(p.id) + '\')">' + t('providers.health') + '</button>' +
                '<button class="btn btn-danger btn-sm" onclick="deleteProvider(\'' + jsEsc(p.id) + '\')">' + t('providers.delete') + '</button>' +
            '</div>' +
        '</div>';
    }).join('');
}

async function deleteProviderModel(providerId, modelId) {
    if (!confirm(t('models.deleteConfirm', {model: modelId}))) return;
    try {
        await api('/admin/providers/' + encodeURIComponent(providerId) +
                  '/models/' + encodeURIComponent(modelId), { method: 'DELETE' });
        await Promise.all([loadProviders(), loadModels()]);
    } catch (e) { toast(t('models.deleteFail') + ': ' + e.message, 'error'); }
}

function addCustomModel(providerId) {
    var provider = providers.find(function(item) { return item.id === providerId; });
    if (!provider) return;
    document.getElementById('modalContent').innerHTML =
        '<h2>' + t('models.addCustomTitle') + '</h2>' +
        '<div class="form-group"><label>' + t('models.modelId') + '</label>' +
            '<input type="text" id="customModelId" placeholder="gpt-4o-preview-1120"></div>' +
        '<div class="form-group"><label>' + t('models.modelName') + '</label>' +
            '<input type="text" id="customModelName" placeholder=""></div>' +
        '<div class="modal-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('common.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="submitCustomModel(\'' + jsEsc(providerId) + '\')">' + t('common.save') + '</button>' +
        '</div>';
    document.getElementById('modal').style.display = 'flex';
}

async function submitCustomModel(providerId) {
    var provider = providers.find(function(item) { return item.id === providerId; });
    if (!provider) return;
    var idInput = document.getElementById('customModelId');
    var nameInput = document.getElementById('customModelName');
    var modelId = (idInput.value || '').trim();
    if (!modelId) { toast(t('models.modelId'), 'error'); return; }
    var modelName = (nameInput.value || '').trim() || modelId;

    var existing = (provider.models || []).map(function(m) {
        return { id: m.id, name: m.name, enabled: m.enabled };
    });
    if (existing.some(function(m) { return m.id === modelId; })) {
        toast(t('models.modelId') + ': ' + modelId, 'error');
        return;
    }
    existing.push({ id: modelId, name: modelName, enabled: true });

    try {
        await api('/admin/providers/' + encodeURIComponent(providerId), {
            method: 'PUT',
            body: JSON.stringify({ models: existing })
        });
        closeModal();
        await Promise.all([loadProviders(), loadModels()]);
    } catch (e) { toast(t('providers.updateFail') + ': ' + e.message, 'error'); }
}

function editModel(providerId, modelId) {
    var m = models.find(function(item) { return item.id === modelId && item.provider === providerId; });
    if (!m) return;
    var isManual = m.source === 'manual';
    var checkbox = '<label><input type="checkbox" id="modelMakeManual" value="1"' +
        (isManual ? ' checked disabled' : '') + '> ' + t('models.makeManual') + '</label>';
    document.getElementById('modalContent').innerHTML =
        '<h2>' + t('models.editModel') + '</h2>' +
        '<div class="form-group"><label>' + t('models.modelId') + '</label>' +
            '<input type="text" value="' + escHtml(modelId) + '" disabled></div>' +
        '<div class="form-group"><label>' + t('models.modelName') + '</label>' +
            '<input type="text" id="modelEditName" value="' + escHtml(m.name || m.id) + '"></div>' +
        '<div class="form-group">' + checkbox + '</div>' +
        '<div class="modal-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('common.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="saveModelEdit(\'' + jsEsc(providerId) + '\', \'' + jsEsc(modelId) + '\')">' + t('models.saveEdit') + '</button>' +
        '</div>';
    document.getElementById('modal').style.display = 'flex';
}

async function saveModelEdit(providerId, modelId) {
    var m = models.find(function(item) { return item.id === modelId && item.provider === providerId; });
    if (!m) return;
    var nameInput = document.getElementById('modelEditName');
    var makeManualCheckbox = document.getElementById('modelMakeManual');
    var newName = (nameInput.value || '').trim() || modelId;
    var makeManual = makeManualCheckbox && makeManualCheckbox.checked && m.source !== 'manual';

    // Build the model update payload: preserve all existing fields, override name/enabled,
    // and add source: 'manual' only when the checkbox is checked.
    var provider = providers.find(function(item) { return item.id === providerId; });
    if (!provider) return;
    var existing = (provider.models || []).map(function(mm) {
        var obj = { id: mm.id, name: mm.name, enabled: mm.enabled };
        if (mm.id === modelId && makeManual) {
            obj.source = 'manual';
        }
        return obj;
    });
    // Update the name for the edited model
    for (var i = 0; i < existing.length; i++) {
        if (existing[i].id === modelId) {
            existing[i].name = newName;
        }
    }

    try {
        await api('/admin/providers/' + encodeURIComponent(providerId), {
            method: 'PUT',
            body: JSON.stringify({ models: existing })
        });
        closeModal();
        await Promise.all([loadProviders(), loadModels()]);
    } catch (e) { toast(t('providers.updateFail') + ': ' + e.message, 'error'); }
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
        '<div class="form-row">' +
            '<div class="form-group"><label>' + t('providers.requestTimeout') + '</label>' +
                '<input type="number" id="providerRequestTimeout" value="' + escHtml(provider.request_timeout || 120) + '" min="1" max="3600"></div>' +
            '<div class="form-group"><label>' + t('providers.retryCount') + '</label>' +
                '<input type="number" id="providerRetryCount" value="' + escHtml(provider.retry_count || 0) + '" min="0" max="10"></div>' +
            '<div class="form-group"><label>' + t('providers.retryBackoff') + '</label>' +
                '<input type="number" id="providerRetryBackoff" value="' + escHtml(provider.retry_backoff == null ? 0.5 : provider.retry_backoff) + '" min="0" max="60" step="0.1"></div>' +
        '</div>' +
        '<div class="form-group"><label><input type="checkbox" id="providerEnabled"' + (provider.enabled === false ? '' : ' checked') + '> ' + t('providers.enabled') + '</label></div>' +
        '<div class="form-group"><label><input type="checkbox" id="providerForceChatCompletions"' + (provider.force_chat_completions ? ' checked' : '') + '> ' + t('providers.forceChatCompletions') + '</label></div>' +
        '<div class="form-group"><label>' + t('providers.extraHeaders') + '</label>' +
            '<textarea id="providerExtraHeaders" rows="3" style="font-family:monospace;font-size:12px" placeholder=\'{"thinking": "enabled"}\'>' + escHtml(JSON.stringify(provider.extra_headers || {}, null, 2)) + '</textarea></div>' +
        '<div class="form-actions">' +
            '<button class="btn btn-secondary" onclick="closeModal()">' + t('common.cancel') + '</button>' +
            '<button class="btn btn-primary" onclick="' + submitAction + '">' + t('common.save') + '</button></div>';
}

function readProviderForm() {
    var eh = document.getElementById('providerExtraHeaders').value.trim();
    var extraHeaders = {};
    if (eh) { try { extraHeaders = JSON.parse(eh); } catch(e) { toast('extra_headers JSON invalid: ' + e.message, 'error'); } }
    return {
        id: document.getElementById('providerId').value.trim(),
        name: document.getElementById('providerName').value.trim(),
        provider_type: document.getElementById('providerType').value,
        api_base: document.getElementById('providerApiBase').value.trim(),
        api_key: document.getElementById('providerApiKey').value.trim(),
        enabled: document.getElementById('providerEnabled').checked,
        extra_headers: extraHeaders,
        request_timeout: parseInt(document.getElementById('providerRequestTimeout').value, 10) || 120,
        retry_count: parseInt(document.getElementById('providerRetryCount').value, 10) || 0,
        retry_backoff: parseFloat(document.getElementById('providerRetryBackoff').value) || 0
        ,force_chat_completions: document.getElementById('providerForceChatCompletions').checked
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

async function checkProviderHealth(id) {
    try {
        var result = await api('/admin/providers/' + encodeURIComponent(id) + '/health');
        if (result.ok) {
            toast(t('providers.healthOk', {n: result.model_count || 0, ms: result.latency_ms || 0}), 'success');
        } else {
            toast(t('providers.healthFail') + ': ' + (result.error || result.status || id), 'error');
        }
    } catch (e) { toast(t('providers.healthFail') + ': ' + e.message, 'error'); }
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
            var isManual = m.source === 'manual';
            var sourceBadge = '<span class="model-source-badge ' + (isManual ? 'manual' : 'auto') + '">' +
                (isManual ? t('models.sourceManual') : t('models.sourceAuto')) + '</span>';
            html += '<div class="model-item">' +
                '<div class="model-info">' +
                    '<span class="model-name">' + escHtml(m.name || m.id) + sourceBadge + '</span>' +
                    '<span class="model-id mono">' + escHtml(m.id) + '</span>' +
                '</div>' +
                '<div class="model-actions">' +
                    '<button class="btn btn-secondary btn-sm" onclick="copyText(\'' + jsEsc(m.id) + '\')">' + t('models.copyId') + '</button>' +
                    '<button class="btn btn-secondary btn-sm" onclick="editModel(\'' + jsEsc(m.provider) + '\', \'' + jsEsc(m.id) + '\')">' + t('common.edit') + '</button>' +
                    '<button class="btn btn-primary btn-sm" onclick="testModel(\'' + jsEsc(m.id) + '\', this)">' + t('models.test') + '</button>' +
                    '<button class="btn btn-danger btn-sm" onclick="deleteProviderModel(\'' + jsEsc(m.provider) + '\', \'' + jsEsc(m.id) + '\')">' + t('common.delete') + '</button>' +
                '</div>' +
            '</div>';
        }

        html += '</div></div>';
    }

    container.innerHTML = html;
}

async function testModel(modelId, btn) {
    var oldText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = t('models.testRunning'); }
    try {
        var result = await api('/admin/models/test', {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId })
        });
        showTestResult(t('testResult.title') + ' - ' + modelId, result);
    } catch (e) {
        toast(t('models.testFail') + ': ' + e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = oldText || t('models.test'); }
    }
}

/* ═══════════════════════════════ Preprocessors ═══════════════════════════════ */

var preprocessorsData = { preprocessors: {}, models: [] };

var imageGenerationData = { generators: {}, models: [], providers: [] };
var comfyWorkflowAnalysis = null;

async function loadImageGeneration() {
    try {
        var data = await api('/admin/image-generation');
        imageGenerationData = { generators: data.generators || {}, models: data.models || [], providers: data.providers || [] };
        renderImageGeneration();
    } catch (e) { toast(t('imageGeneration.loadFail') + ': ' + e.message, 'error'); }
}

function renderImageGeneration() {
    var container = document.getElementById('imageGenerationContent');
    var ids = Object.keys(imageGenerationData.generators);
    var active = ids.length ? imageGenerationData.generators[ids[0]] : {};
    var html = '<div class="preprocessors-layout">';
    var providerOptions = '<option value="">' + t('imageGeneration.selectModel') + '</option>';
    (imageGenerationData.providers || []).forEach(function(provider) {
        (provider.models || []).forEach(function(model) {
            // Native optgroup labels are unreadable in some Windows dark-theme
            // dropdowns. Keep the exact composite ID visible on every option so
            // duplicate model names from different providers are unambiguous.
            providerOptions += '<option value="' + escHtml(model.provider_model) + '">' + escHtml(model.provider_model) + '</option>';
        });
    });
    html += '<div class="preprocessor-config-col"><div class="section-sub-header"><h3>' + t('imageGeneration.config') + '</h3></div>';
    html += '<div class="preprocessor-card glass"><div class="preprocessor-card-body">' +
        '<div class="form-group"><label>' + t('imageGeneration.backendType') + '</label><select id="imageBackendType" onchange="updateImageBackendFields()"><option value="existing_model">' + t('imageGeneration.existingModel') + '</option><option value="external_model">' + t('imageGeneration.externalModel') + '</option><option value="comfyui">' + t('imageGeneration.comfyui') + '</option></select></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.providerModel') + '</label><select id="imageProviderModel">' + providerOptions + '</select></div>' +
        '<div id="imageExternalFields" style="display:none">' +
        '<div class="form-group"><label>' + t('imageGeneration.apiBase') + '</label><input id="imageApiBase"></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.apiKey') + '</label><input type="password" id="imageApiKey"></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.model') + '</label><input id="imageModel"></div></div>' +
        '<div id="imageComfyFields" style="display:none">' +
        '<div class="form-group"><label>' + t('imageGeneration.comfyBase') + '</label><input id="imageComfyBase" placeholder="http://127.0.0.1:8188"></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.apiKey') + '</label><input type="password" id="imageComfyApiKey"></div>' +
        '<div class="form-actions compact comfy-workflow-actions"><button type="button" class="btn btn-secondary" id="imageFetchWorkflowsBtn" onclick="fetchComfyWorkflows(this)">' + t('imageGeneration.fetchWorkflows') + '</button></div>' +
        '<div id="imageSavedWorkflowFields" style="display:none"><div class="form-group"><label>' + t('imageGeneration.savedWorkflow') + '</label><select id="imageSavedWorkflow"><option value="">' + t('imageGeneration.selectWorkflow') + '</option></select></div><div class="form-actions compact"><button type="button" class="btn btn-secondary" onclick="loadSelectedComfyWorkflow()">' + t('imageGeneration.loadWorkflow') + '</button></div></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.workflow') + '</label><textarea id="imageComfyWorkflow" class="image-workflow-json" spellcheck="false"></textarea><div class="form-hint">' + t('imageGeneration.workflowHint') + '</div></div>' +
        '<div class="form-actions compact"><button type="button" class="btn btn-secondary" id="imageWorkflowAnalyzeBtn" onclick="analyzeComfyWorkflow(this)">' + t('imageGeneration.analyzeWorkflow') + '</button></div>' +
        '<div id="imageComfyMappings"></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.pollInterval') + '</label><input type="number" id="imageComfyPollInterval" value="1" min="0.2" max="10" step="0.1"></div></div>' +
        '<div class="form-group"><label>' + t('imageGeneration.timeout') + '</label><input type="number" id="imageTimeout" value="' + (active.timeout || 180) + '" min="1" max="3600"></div>' +
        '<div class="form-group"><label><input type="checkbox" id="imageEnabled"' + (active.enabled === false ? '' : ' checked') + '> ' + t('imageGeneration.enabled') + '</label></div>' +
        '<div class="form-actions"><button class="btn btn-secondary" id="imageGenerationTestBtn" onclick="testImageGeneration(this)">' + t('imageGeneration.test') + '</button><button class="btn btn-primary" onclick="saveImageGeneration()">' + t('imageGeneration.save') + '</button></div>' +
        '</div></div></div>';
    html += '<div class="preprocessor-models-col"><div class="section-sub-header"><h3>' + t('imageGeneration.models') + '</h3></div>';
    if (!imageGenerationData.models.length) {
        html += '<div class="empty-state"><div class="empty-icon">&#128269;</div><p>' + t('preprocessors.modelsEmpty') + '</p></div>';
    } else {
        var groupedModels = {};
        imageGenerationData.models.forEach(function(model) {
            var providerKey = model.provider_id || model.provider_name || '-';
            if (!groupedModels[providerKey]) {
                groupedModels[providerKey] = {
                    name: model.provider_name || model.provider_id || '-',
                    models: []
                };
            }
            groupedModels[providerKey].models.push(model);
        });
        Object.keys(groupedModels).sort(function(left, right) {
            return groupedModels[left].name.localeCompare(groupedModels[right].name);
        }).forEach(function(providerKey) {
            var group = groupedModels[providerKey];
            html += '<div class="model-group glass">' +
                '<div class="model-group-header">' +
                    '<span class="model-group-title">' + escHtml(group.name) + '</span>' +
                    '<span class="model-group-count">' + group.models.length + ' ' + t('models.count') + '</span>' +
                '</div><div class="model-group-list">';
            group.models.forEach(function(model) {
                html += '<div class="model-toggle-item">' +
                    '<div class="model-toggle-info">' +
                        '<span class="model-toggle-name">' + escHtml(model.model_id) + '</span>' +
                        '<span class="model-toggle-status ' + (model.image_generation ? 'on' : 'off') + '">' +
                            (model.image_generation ? t('imageGeneration.on') : t('imageGeneration.off')) +
                        '</span>' +
                    '</div><label class="toggle-switch">' +
                        '<input type="checkbox" ' + (model.image_generation ? 'checked' : '') +
                            ' onchange="toggleModelImageGeneration(\'' + jsEsc(model.provider_model) + '\', this.checked)">' +
                        '<span class="toggle-slider"></span></label></div>';
            });
            html += '</div></div>';
        });
    }
    html += '</div></div>';
    container.innerHTML = html;
    if (active.backend_type) document.getElementById('imageBackendType').value = active.backend_type === 'openai_images' ? 'existing_model' : active.backend_type;
    if (active.provider_model) document.getElementById('imageProviderModel').value = active.provider_model;
    document.getElementById('imageApiBase').value = active.api_base || '';
    document.getElementById('imageApiKey').value = '';
    document.getElementById('imageApiKey').placeholder = active.has_api_key ? '********' : '';
    document.getElementById('imageModel').value = active.model || '';
    document.getElementById('imageComfyBase').value = active.backend_type === 'comfyui' ? (active.api_base || '') : '';
    document.getElementById('imageComfyApiKey').value = '';
    document.getElementById('imageComfyApiKey').placeholder = active.backend_type === 'comfyui' && active.has_api_key ? '********' : '';
    document.getElementById('imageComfyWorkflow').value = active.workflow && Object.keys(active.workflow).length ? JSON.stringify(active.workflow, null, 2) : '';
    document.getElementById('imageComfyPollInterval').value = active.poll_interval || 1;
    if (active.workflow && Object.keys(active.workflow).length) {
        comfyWorkflowAnalysis = buildLocalComfyAnalysis(active.workflow, active.workflow_mapping || {});
        renderComfyMappings(active.workflow_mapping || {});
    } else {
        comfyWorkflowAnalysis = null;
        renderComfyMappings({});
    }
    updateImageBackendFields();
}

function updateImageBackendFields() {
    var type = document.getElementById('imageBackendType').value;
    document.getElementById('imageProviderModel').closest('.form-group').style.display = type === 'existing_model' ? '' : 'none';
    document.getElementById('imageExternalFields').style.display = type === 'external_model' ? '' : 'none';
    document.getElementById('imageComfyFields').style.display = type === 'comfyui' ? '' : 'none';
}

async function fetchComfyWorkflows(btn) {
    var old = btn ? btn.textContent : '';
    try {
        if (btn) { btn.disabled = true; btn.textContent = t('imageGeneration.fetchingWorkflows'); }
        var result = await api('/admin/image-generation/comfyui/workflows', { method: 'POST', body: JSON.stringify({ api_base: document.getElementById('imageComfyBase').value.trim(), api_key: document.getElementById('imageComfyApiKey').value.trim(), timeout: 30 }) });
        var select = document.getElementById('imageSavedWorkflow');
        select.innerHTML = '<option value="">' + t('imageGeneration.selectWorkflow') + '</option>' + (result.workflows || []).map(function(name) { return '<option value="' + escHtml(name) + '">' + escHtml(name.replace(/\.json$/i, '')) + '</option>'; }).join('');
        document.getElementById('imageSavedWorkflowFields').style.display = '';
    } catch (e) { toast(t('imageGeneration.fetchWorkflowsFail') + ': ' + e.message, 'error'); }
    finally { if (btn) { btn.disabled = false; btn.textContent = old || t('imageGeneration.fetchWorkflows'); } }
}

async function loadSelectedComfyWorkflow() {
    try {
        var name = document.getElementById('imageSavedWorkflow').value;
        if (!name) return;
        var result = await api('/admin/image-generation/comfyui/load-workflow', { method: 'POST', body: JSON.stringify({ api_base: document.getElementById('imageComfyBase').value.trim(), api_key: document.getElementById('imageComfyApiKey').value.trim(), workflow_name: name, timeout: 30 }) });
        document.getElementById('imageComfyWorkflow').value = JSON.stringify(result.workflow || {}, null, 2);
        comfyWorkflowAnalysis = null; renderComfyMappings({});
        toast(t('imageGeneration.workflowLoaded'), 'success');
    } catch (e) { toast(t('imageGeneration.fetchWorkflowsFail') + ': ' + e.message, 'error'); }
}

function buildLocalComfyAnalysis(workflow, mapping) {
    var candidates = { prompt: [], negative_prompt: [], width: [], height: [], seed: [], steps: [], cfg: [], batch_size: [] };
    var outputs = [];
    Object.keys(workflow || {}).forEach(function(nodeId) {
        var node = workflow[nodeId] || {};
        var label = nodeId + ' · ' + (((node._meta || {}).title) || node.class_type || 'Node') + ' (' + (node.class_type || '?') + ')';
        if (/saveimage|previewimage|output|save_image/i.test(node.class_type || '')) outputs.push({ node_id: nodeId, label: label });
        Object.keys(node.inputs || {}).forEach(function(input) {
            var value = node.inputs[input];
            if (typeof value === 'string') {
                var item = { node_id: nodeId, input: input, label: label + ' → ' + input };
                candidates.prompt.push(item); candidates.negative_prompt.push(item);
            }
            var normalized = ({ noise_seed: 'seed', cfg_scale: 'cfg', batch: 'batch_size' })[input.toLowerCase()] || input.toLowerCase();
            if (candidates[normalized] && (typeof value === 'number' || typeof value === 'string')) candidates[normalized].push({ node_id: nodeId, input: input, label: label + ' → ' + input });
        });
    });
    return { workflow: workflow, candidates: candidates, outputs: outputs, suggestions: mapping || {} };
}

function mappingValue(item) { return item ? JSON.stringify([item.node_id, item.input]) : ''; }
function mappingOptions(items, selected) {
    var selectedValue = mappingValue(selected);
    var html = '<option value="">' + t('imageGeneration.selectMapping') + '</option>';
    (items || []).forEach(function(item) {
        var value = mappingValue(item);
        html += '<option value="' + escHtml(value) + '"' + (value === selectedValue ? ' selected' : '') + '>' + escHtml(item.label) + '</option>';
    });
    return html;
}

function renderComfyMappings(selected) {
    var container = document.getElementById('imageComfyMappings');
    if (!container) return;
    if (!comfyWorkflowAnalysis) { container.innerHTML = ''; return; }
    selected = selected || comfyWorkflowAnalysis.suggestions || {};
    var fields = [
        ['prompt', 'positivePrompt'], ['negative_prompt', 'negativePrompt'], ['width', 'width'],
        ['height', 'height'], ['seed', 'seed'], ['steps', 'steps'], ['cfg', 'cfg'], ['batch_size', 'batchSize']
    ];
    var html = '<div class="comfy-mapping-panel"><h4>' + t('imageGeneration.mapping') + '</h4><div class="comfy-mapping-grid">';
    fields.forEach(function(pair) {
        var choice = selected[pair[0]] || (comfyWorkflowAnalysis.suggestions || {})[pair[0]];
        html += '<div class="form-group"><label>' + t('imageGeneration.' + pair[1]) + '</label><select data-comfy-mapping="' + pair[0] + '">' + mappingOptions(comfyWorkflowAnalysis.candidates[pair[0]], choice) + '</select></div>';
    });
    var outputSelected = selected.output_node_id || (comfyWorkflowAnalysis.suggestions || {}).output_node_id || '';
    html += '<div class="form-group"><label>' + t('imageGeneration.outputNode') + '</label><select id="imageComfyOutputNode"><option value="">' + t('imageGeneration.autoOutput') + '</option>';
    (comfyWorkflowAnalysis.outputs || []).forEach(function(item) { html += '<option value="' + escHtml(item.node_id) + '"' + (item.node_id === outputSelected ? ' selected' : '') + '>' + escHtml(item.label) + '</option>'; });
    html += '</select></div></div></div>';
    container.innerHTML = html;
}

async function analyzeComfyWorkflow(btn) {
    var old = btn ? btn.textContent : '';
    try {
        var workflow = JSON.parse(document.getElementById('imageComfyWorkflow').value || '{}');
        if (btn) { btn.disabled = true; btn.textContent = t('imageGeneration.analyzingWorkflow'); }
        comfyWorkflowAnalysis = await api('/admin/image-generation/comfyui/analyze-workflow', { method: 'POST', body: JSON.stringify({ workflow: workflow }) });
        document.getElementById('imageComfyWorkflow').value = JSON.stringify(comfyWorkflowAnalysis.workflow || workflow, null, 2);
        renderComfyMappings(comfyWorkflowAnalysis.suggestions || {});
        toast(t('imageGeneration.workflowAnalyzed') + ': ' + comfyWorkflowAnalysis.node_count, 'success');
        if (comfyWorkflowAnalysis.converted) toast(t('imageGeneration.workflowConverted'), 'info');
    } catch (e) { toast(t('imageGeneration.workflowAnalyzeFail') + ': ' + e.message, 'error'); }
    finally { if (btn) { btn.disabled = false; btn.textContent = old || t('imageGeneration.analyzeWorkflow'); } }
}

function collectComfyMapping() {
    var result = {};
    document.querySelectorAll('[data-comfy-mapping]').forEach(function(select) {
        if (!select.value) return;
        var parts = JSON.parse(select.value);
        result[select.getAttribute('data-comfy-mapping')] = { node_id: parts[0], input: parts[1] };
    });
    var output = document.getElementById('imageComfyOutputNode');
    result.output_node_id = output ? output.value : '';
    return result;
}

async function saveImageGeneration() {
    var id = Object.keys(imageGenerationData.generators)[0] || 'default';
    try {
        var type = document.getElementById('imageBackendType').value;
        var config = { backend_type: type, provider_model: type === 'existing_model' ? document.getElementById('imageProviderModel').value : '', api_base: type === 'external_model' ? document.getElementById('imageApiBase').value.trim() : (type === 'comfyui' ? document.getElementById('imageComfyBase').value.trim() : ''), api_key: type === 'external_model' ? document.getElementById('imageApiKey').value.trim() : (type === 'comfyui' ? document.getElementById('imageComfyApiKey').value.trim() : ''), model: type === 'external_model' ? document.getElementById('imageModel').value.trim() : '', timeout: parseInt(document.getElementById('imageTimeout').value) || 180, enabled: document.getElementById('imageEnabled').checked };
        if (type === 'comfyui') {
            config.workflow = JSON.parse(document.getElementById('imageComfyWorkflow').value || '{}');
            config.workflow_mapping = collectComfyMapping();
            config.poll_interval = parseFloat(document.getElementById('imageComfyPollInterval').value) || 1;
        }
        await api('/admin/image-generation/' + encodeURIComponent(id), { method: 'PUT', body: JSON.stringify(config) });
        toast(t('common.saved') || t('imageGeneration.save'), 'success'); loadImageGeneration();
    } catch (e) { toast(t('imageGeneration.saveFail') + ': ' + e.message, 'error'); }
}

async function testImageGeneration(btn) {
    var id = Object.keys(imageGenerationData.generators)[0] || '';
    if (!id) { toast(t('imageGeneration.noGenerator'), 'error'); return; }
    var oldText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = t('imageGeneration.testRunning'); }
    toast(t('imageGeneration.testStarted'), 'info');
    try {
        var result = await api('/admin/image-generation/test', {
            method: 'POST',
            body: JSON.stringify({ generator_id: id })
        });
        showTestResult(t('testResult.title') + ' - ' + t('imageGeneration.title'), result);
    } catch (e) {
        toast(t('imageGeneration.testFail') + ': ' + e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = oldText || t('imageGeneration.test'); }
    }
}

async function toggleModelImageGeneration(providerModel, enabled) {
    try { await api('/admin/models/image-generation', { method: 'PUT', body: JSON.stringify({ model_id: providerModel, enabled: enabled }) }); loadImageGeneration(); }
    catch (e) { toast(t('imageGeneration.toggleFail') + ': ' + e.message, 'error'); }
}

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
                '<button class="btn btn-primary btn-sm" onclick="testPreprocessor(\'' + jsEsc(id) + '\', this)">' + t('preprocessors.test') + '</button>' +
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
                '<span>' + (p.timeout || 120) + 's</span>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.maxImages') + ':</span>' +
                '<span>' + (p.max_images || 1) + '</span>' +
            '</div>' +
            '<div class="preprocessor-field">' +
                '<span class="preprocessor-label">' + t('preprocessors.maxTokens') + ':</span>' +
                '<span>' + (p.max_tokens || 2048) + '</span>' +
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
            '<input type="text" id="prepModel" list="prepModelList" value="' + escHtml(preprocessor.model || '') + '" placeholder="' + t('preprocessors.modelPlaceholder') + '" style="flex:1" autocomplete="off">' +
            '<datalist id="prepModelList"></datalist>' +
            '<button class="btn btn-secondary btn-sm" type="button" onclick="fetchPreprocessorModels()">' + t('preprocessors.fetchModels') + '</button>' +
            '</div></div>' +
        '<div class="form-group"><label>' + t('preprocessors.timeout') + '</label>' +
            '<input type="number" id="prepTimeout" value="' + (preprocessor.timeout || 120) + '" min="1" max="300"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.maxImages') + '</label>' +
            '<input type="number" id="prepMaxImages" value="' + (preprocessor.max_images || 10) + '" min="1" max="50"></div>' +
        '<div class="form-group"><label>' + t('preprocessors.maxTokens') + '</label>' +
            '<input type="number" id="prepMaxTokens" value="' + (preprocessor.max_tokens || 2048) + '" min="128" max="8192"></div>' +
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

async function testPreprocessor(id, btn) {
    var oldText = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = t('preprocessors.testRunning'); }
    try {
        var result = await api('/admin/preprocessors/test', {
            method: 'POST',
            body: JSON.stringify({ preprocessor_id: id })
        });
        showTestResult(t('testResult.title') + ' - ' + id, result);
    } catch (e) {
        toast(t('preprocessors.testFail') + ': ' + e.message, 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = oldText || t('preprocessors.test'); }
    }
}

function showTestResult(title, result) {
    result = result || {};
    var statusText = result.status === 'ok' ? t('testResult.ok') : t('testResult.fail');
    var statusClass = result.status === 'ok' ? 'badge-ok' : 'badge-fail';
    var latency = result.latency_ms === '' || result.latency_ms === null ? '-' : result.latency_ms + 'ms';
    var rows = [
        '<div class="detail-kv"><div class="detail-label">' + escHtml(t('testResult.status')) + '</div><div class="detail-value"><span class="status-badge ' + statusClass + '">' + escHtml(statusText) + '</span></div></div>',
        detailRow(t('testResult.latency'), latency),
        detailRow(t('testResult.model'), result.model || result.preprocessor_id || '-'),
        detailRow(t('testResult.provider'), result.provider_id || result.provider_type || '-'),
        detailRow(t('testResult.preview'), result.preview || '-'),
        detailRow(t('testResult.error'), result.error || '-'),
        detailRow(t('testResult.usage'), result.usage || {})
    ];
    document.getElementById('modalContent').innerHTML = '<h2>' + escHtml(title) + '</h2>' +
        '<div class="detail-section"><div class="detail-grid">' + rows.join('') + '</div></div>' +
        '<div class="form-actions"><button class="btn btn-primary" onclick="closeModal()">OK</button></div>';
    document.getElementById('modal').style.display = 'flex';
}

function readPreprocessorForm() {
    return {
        _name: document.getElementById('prepName').value.trim(),
        api_base: document.getElementById('prepApiBase').value.trim(),
        model: document.getElementById('prepModel').value.trim(),
        api_key: document.getElementById('prepApiKey').value.trim(),
        timeout: parseInt(document.getElementById('prepTimeout').value) || 120,
        max_images: parseInt(document.getElementById('prepMaxImages').value) || 10,
        max_tokens: parseInt(document.getElementById('prepMaxTokens').value) || 2048,
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
var _historyCharts = [];
var _statsTab = 'realtime'; // 'realtime' or 'history'
var _realtimeTrendMode = 'calls'; // 'calls' or 'tokens' — survives polling refreshes
var _CHART_COLORS = ['#818cf8','#34d399','#fbbf24','#f472b6','#f87171','#38bdf8','#a78bfa','#fb923c','#6366f1','#4ade80',
    '#ec4899','#14b8a6','#f59e0b','#6366f1','#10b981','#e11d48','#0ea5e9','#d946ef','#84cc16','#f97316'];

function _destroyCharts() {
    for (var i = 0; i < _statsCharts.length; i++) {
        try { _statsCharts[i].destroy(); } catch (e) {}
    }
    _statsCharts = [];
}

function _destroyHistoryCharts() {
    for (var i = 0; i < _historyCharts.length; i++) {
        try { _historyCharts[i].destroy(); } catch (e) {}
    }
    _historyCharts = [];
}

/* Build or update a per-model stacked bar chart for trend visualization.
   canvasId  — <canvas> element id
   tmData    — { labels, models, calls, tokens }
   mode      — 'calls' or 'tokens'
   chartArr  — array to push the Chart instance into
   existing  — if provided, update this chart in-place instead of creating new */
/* Return true if every value in arr is 0 */
function _isAllZero(arr) {
    for (var i = 0; i < arr.length; i++) { if (arr[i] !== 0) return false; }
    return !!arr.length;
}

function _buildOrupdateTrendChart(canvasId, tmData, mode, chartArr, existing) {
    var data = mode === 'tokens' ? tmData.tokens : tmData.calls;
    var useLine = tmData.labels.length > 12;
    var datasets = [];
    for (var i = 0; i < tmData.models.length; i++) {
        if (_isAllZero(data[i])) continue;
        var ds = {
            label: tmData.models[i],
            data: data[i],
            backgroundColor: _CHART_COLORS[i % _CHART_COLORS.length],
        };
        if (useLine) {
            ds.borderColor = _CHART_COLORS[i % _CHART_COLORS.length];
            ds.borderWidth = 2;
            ds.pointRadius = 1;
            ds.pointHoverRadius = 4;
            ds.tension = 0.15;
            ds.fill = false;
        } else {
            ds.borderRadius = 2;
        }
        datasets.push(ds);
    }
    if (existing) {
        var curType = existing.config.type || 'bar';
        var needType = useLine ? 'line' : 'bar';
        if (curType !== needType) {
            existing.destroy();
            var idx = chartArr.indexOf(existing);
            if (idx >= 0) chartArr.splice(idx, 1);
        } else {
            existing.data.labels = tmData.labels;
            existing.data.datasets = datasets;
            existing.options.plugins.title.text = t('stats.trendChart') + ' — ' + (mode === 'tokens' ? t('stats.trendTokens') : t('stats.trendCalls'));
            existing.update();
            return existing;
        }
    }
    var ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    var maxWidth = Math.max(30, Math.min(80, Math.floor(600 / Math.max(tmData.labels.length, 1))));
    var scaleOpts = useLine
        ? { x: { grid: { display: false }, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 }, maxRotation: 45 } },
            y: { beginAtZero: true, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 } } } }
        : { x: { stacked: true, grid: { display: false }, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 }, maxRotation: 45 } },
            y: { stacked: true, beginAtZero: true, ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary').trim(), font: { size: 11 } } } };
    var chart = new Chart(ctx, {
        type: useLine ? 'line' : 'bar',
        data: { labels: tmData.labels, datasets: datasets },
        options: {
            responsive: true, maintainAspectRatio: true,
            barPercentage: 0.85, categoryPercentage: 0.75, maxBarThickness: maxWidth,
            interaction: { mode: 'index', intersect: false },
            scales: scaleOpts,
            plugins: {
                title: { display: true, text: t('stats.trendChart') + ' — ' + (mode === 'tokens' ? t('stats.trendTokens') : t('stats.trendCalls')), font: { size: 14, weight: '700' }, color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(), padding: { bottom: 12 } },
                legend: { position: 'bottom', labels: { usePointStyle: true, pointStyleWidth: 8, color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim(), font: { size: 11 }, padding: 12 }, filter: function(item) { return item.text !== undefined; } },
                tooltip: {
                    mode: 'index', intersect: false,
                    filter: function(item) { return item.raw !== 0; },
                    itemSort: function(a, b) { return b.raw - a.raw; },
                    callbacks: {
                        footer: function(items) {
                            if (!items || items.length === 0) return '';
                            var sum = 0;
                            for (var i = 0; i < items.length; i++) { sum += items[i].raw; }
                            return '∑ ' + sum.toLocaleString();
                        }
                    }
                }
            }
        }
    });
    chartArr.push(chart);
    return chart;
}

/* Toggle trend chart between calls/tokens mode.
   Simply updates datasets and title on the existing chart — no destroy/recreate needed. */
function _switchTrendMode(chartInstanceVar, tmDataVar, mode) {
    var chart = window[chartInstanceVar];
    var tmData = window[tmDataVar];
    if (!chart || !tmData || !tmData.labels || tmData.labels.length === 0) return;
    var data = mode === 'tokens' ? tmData.tokens : tmData.calls;
    var useLine = tmData.labels.length > 12;
    var datasets = [];
    for (var i = 0; i < tmData.models.length; i++) {
        if (_isAllZero(data[i])) continue;
        var ds = {
            label: tmData.models[i],
            data: data[i],
            backgroundColor: _CHART_COLORS[i % _CHART_COLORS.length],
        };
        if (useLine) {
            ds.borderColor = _CHART_COLORS[i % _CHART_COLORS.length];
            ds.borderWidth = 2;
            ds.pointRadius = 1;
            ds.pointHoverRadius = 4;
            ds.tension = 0.15;
            ds.fill = false;
        } else {
            ds.borderRadius = 2;
        }
        datasets.push(ds);
    }
    chart.data.datasets = datasets;
    chart.options.plugins.tooltip.filter = function(item) { return item.raw !== 0; };
    chart.options.plugins.title.text = t('stats.trendChart') + ' — ' + (mode === 'tokens' ? t('stats.trendTokens') : t('stats.trendCalls'));
    chart.update();
}

function switchRealtimeTrend(mode) {
    _realtimeTrendMode = mode;
    document.getElementById('realtimeTrendCalls').className = 'trend-btn' + (mode === 'calls' ? ' active' : '');
    document.getElementById('realtimeTrendTokens').className = 'trend-btn' + (mode === 'tokens' ? ' active' : '');
    _switchTrendMode('_realtimeTrendChart', '_realtimeTmData', mode);
}

function switchHistoryTrend(mode) {
    document.getElementById('historyTrendCalls').className = 'trend-btn' + (mode === 'calls' ? ' active' : '');
    document.getElementById('historyTrendTokens').className = 'trend-btn' + (mode === 'tokens' ? ' active' : '');
    _switchTrendMode('_historyTrendChart', '_historyTmData', mode);
}

function switchStatsTab(tab) {
    _statsTab = tab;
    var realtimeBtn = document.getElementById('statsTabRealtime');
    var historyBtn = document.getElementById('statsTabHistory');
    var realtimePanel = document.getElementById('statsRealtimePanel');
    var historyPanel = document.getElementById('statsHistoryPanel');
    if (realtimeBtn) realtimeBtn.className = tab === 'realtime' ? 'stats-tab active' : 'stats-tab';
    if (historyBtn) historyBtn.className = tab === 'history' ? 'stats-tab active' : 'stats-tab';
    if (realtimePanel) realtimePanel.style.display = tab === 'realtime' ? '' : 'none';
    if (historyPanel) historyPanel.style.display = tab === 'history' ? '' : 'none';
    if (tab === 'history') loadHistoryStats();
}

function _onHistoryKeydown(e) {
    if (e.key === 'Enter') { e.preventDefault(); loadHistoryStats(); }
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
    _destroyHistoryCharts();
}

async function confirmResetStats() {
    if (!confirm(t('stats.resetConfirm'))) return;
    try {
        await api('/admin/stats/reset', { method: 'POST' });
        await loadStats();
    } catch (e) { toast(t('stats.resetFail') + ': ' + e.message, 'error'); }
}

var _lastHistoryData = null;

async function loadHistoryStats() {
    var fromInput = document.getElementById('historyFrom');
    var toInput = document.getElementById('historyTo');
    var granSelect = document.getElementById('historyGranularity');
    if (!fromInput || !toInput || !granSelect) return;
    var fromVal = fromInput.value;
    var toVal = toInput.value;
    var gran = granSelect.value;
    if (!fromVal || !toVal) return;
    var container = document.getElementById('historyContent');
    if (container) container.innerHTML = '<div class="loading-spinner"><span class="spinner"></span><p>' + t('stats.loading') + '</p></div>';
    try {
        var data = await api('/admin/stats/history?from_ts=' + encodeURIComponent(fromVal) + '&to_ts=' + encodeURIComponent(toVal) + '&granularity=' + gran);
        _lastHistoryData = data;
        renderHistoryStats(data);
    } catch (e) {
        if (_lastHistoryData) {
            renderHistoryStats(_lastHistoryData);
        } else if (container) {
            container.innerHTML = '<div class="empty-state"><div class="empty-icon">&#9888;</div><p>' + t('stats.loadFail') + '</p></div>';
        }
        toast(t('stats.loadFail') + ': ' + e.message, 'error');
    }
}

function renderHistoryStats(data) {
    var overall = data.overall || {};
    var models = data.models || [];
    var users = data.users || [];
    var timeline = data.timeline || {};
    var total = overall.total_calls || 0;
    var failed = overall.failed_calls || 0;
    var tokens = overall.total_tokens || 0;
    var successRate = total > 0 ? ((total - failed) / total * 100).toFixed(1) : '100.0';

    // Inline summary line
    var summaryLine = '<div class="history-summary">' +
        '<span class="history-stat"><strong>' + total.toLocaleString() + '</strong> ' + t('stats.periodCalls') + '</span>' +
        '<span class="history-stat success"><strong>' + successRate + '%</strong> ' + t('stats.periodSuccessRate') + '</span>' +
        '<span class="history-stat danger"><strong>' + failed.toLocaleString() + '</strong> ' + t('stats.failedCalls') + '</span>' +
        '<span class="history-stat"><strong>' + tokens.toLocaleString() + '</strong> ' + t('stats.periodTokens') + '</span>' +
        '<span class="history-stat"><strong>' + (overall.image_generation_calls || 0).toLocaleString() + '</strong> ' + t('stats.imageGenerationCalls') + '</span>' +
        '<span class="history-stat"><strong>' + (overall.image_generation_images || 0).toLocaleString() + '</strong> ' + t('stats.generatedImages') + '</span>' +
        '</div>';

    var container = document.getElementById('historyContent');
    if (!container) return;

    _destroyHistoryCharts();
    var hasData = total > 0;

    if (!hasData) {
        container.innerHTML = summaryLine +
            '<div class="empty-state"><div class="empty-icon">&#128202;</div><p>' + t('stats.historyNoData') + '</p><p class="empty-sub">' + t('stats.historyNoDataHint') + '</p></div>';
        return;
    }

    var html = summaryLine;

    // Trend chart
    var htmDataAvail = timeline.labels && timeline.labels.length > 0;
    if (htmDataAvail) {
        html += '<div class="chart-card glass"><div class="trend-header"><h3>' + t('stats.trendChart') + '</h3><div class="trend-toggle"><button class="trend-btn active" id="historyTrendCalls" onclick="switchHistoryTrend(\'calls\')">' + t('stats.trendCalls') + '</button><button class="trend-btn" id="historyTrendTokens" onclick="switchHistoryTrend(\'tokens\')">' + t('stats.trendTokens') + '</button></div></div><div class="chart-wrap"><canvas id="historyTrendChart"></canvas></div></div>';
    }

    // Model breakdown table
    html += '<div class="table-card glass"><h3>' + t('stats.modelBreakdown') + '</h3>';
    html += '<table class="modern-table"><thead><tr><th>' + t('stats.model') + '</th><th>' + t('stats.totalCalls') + '</th><th>' + t('stats.failedCalls') + '</th><th>' + t('stats.tokens') + '</th></tr></thead><tbody>';
    for (var i = 0; i < models.length; i++) {
        var m = models[i];
        html += '<tr><td>' + escHtml(m.model) + '</td><td>' + m.total.toLocaleString() + '</td><td>' + m.failed.toLocaleString() + '</td><td>' + m.tokens.toLocaleString() + '</td></tr>';
    }
    if (models.length === 0) {
        html += '<tr><td colspan="4" style="text-align:center;color:var(--text-tertiary);padding:16px">' + t('stats.noRecords') + '</td></tr>';
    }
    html += '</tbody></table></div>';

    // User breakdown table
    html += '<div class="table-card glass"><h3>' + t('stats.userBreakdown') + '</h3>';
    html += '<table class="modern-table"><thead><tr><th>' + t('stats.client') + '</th><th>' + t('stats.totalCalls') + '</th><th>' + t('stats.failedCalls') + '</th><th>' + t('stats.tokens') + '</th></tr></thead><tbody>';
    for (var j = 0; j < users.length; j++) {
        var u = users[j];
        html += '<tr><td>' + escHtml(u.username) + '</td><td>' + u.total.toLocaleString() + '</td><td>' + u.failed.toLocaleString() + '</td><td>' + u.tokens.toLocaleString() + '</td></tr>';
    }
    if (users.length === 0) {
        html += '<tr><td colspan="4" style="text-align:center;color:var(--text-tertiary);padding:16px">' + t('stats.noRecords') + '</td></tr>';
    }
    html += '</tbody></table></div>';

    container.innerHTML = html;

    // Create trend chart
    if (timeline.labels && timeline.labels.length > 0) {
        var tmData = data.timeline_models || null;
        if (tmData && tmData.models && tmData.models.length > 0) {
            window._historyTmData = tmData;
            var hChart = _buildOrupdateTrendChart('historyTrendChart', tmData, 'calls', _historyCharts, null);
            window._historyTrendChart = hChart;
        } else {
            window._historyTmData = null;
            window._historyTrendChart = null;
            var trendCtx = document.getElementById('historyTrendChart');
            if (trendCtx) {
                _historyCharts.push(new Chart(trendCtx, {
                    type: 'bar',
                    data: { labels: timeline.labels, datasets: [
                        { label: t('stats.chartSuccess'), data: timeline.total.map(function(v, idx) { return v - (timeline.failed[idx] || 0); }), backgroundColor: '#34d399', borderRadius: 4 },
                        { label: t('stats.chartFail'), data: timeline.failed, backgroundColor: '#f87171', borderRadius: 4 }
                    ] },
                    options: { responsive: true, maintainAspectRatio: true, scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } }
                }));
            }
        }
    } else {
        window._historyTmData = null;
        window._historyTrendChart = null;
    }
}

function _buildRealtimePanel(stats) {
    var activeModels = Object.keys(stats.stats_by_model || {}).length;
    var hasData = stats.total_calls > 0;
    var log = stats.request_log || [];
    var tableHTML = '<div class="table-card glass"><h3>' + t('stats.realtime') + '</h3>' +
        '<table class="modern-table"><thead><tr>' +
        '<th>' + t('stats.time') + '</th><th>' + t('stats.requestType') + '</th><th>' + t('stats.client') + '</th><th>' + t('stats.key') + '</th><th>' + t('stats.requestedModel') + '</th><th>' + t('stats.routedModel') + '</th><th>' + t('stats.model') + '</th><th>' + t('stats.endpoint') + '</th><th>' + t('stats.tokens') + '</th><th>' + t('stats.status') + '</th><th>' + (t('stats.details') || 'Details') + '</th>' +
        '</tr></thead><tbody>';

    var displayLog = log.slice(0, 40);
    window._requestLogDetails = displayLog;
    for (var i = 0; i < displayLog.length; i++) {
        var entry = displayLog[i];
        var badge = requestStatusClass(entry);
        var statusLabel = requestStatusLabel(entry);
        var reqModel = entry.requested_model || entry.model;
        var details = entry.details || {};
        var routedModel = entry.routed_model || details.routed_model || '';
        var isImageRequest = entry.request_kind === 'image_generation' || details.request_kind === 'image_generation';
        var isDegraded = entry.status === 'degraded' || (!isImageRequest && (entry.fallback_status === 'used' || details.fallback_status === 'used'));
        var modelChanged = reqModel && entry.model && reqModel !== entry.model;
        tableHTML += '<tr>' +
            '<td>' + escHtml(entry.time) + '</td>' +
            '<td><span class="badge ' + requestKindClass(entry) + '">' + escHtml(requestKindLabel(entry)) + '</span></td>' +
            '<td>' + escHtml(entry.username) + '</td>' +
            '<td class="mono">' + escHtml(entry.api_key) + '</td>' +
            '<td>' + escHtml(reqModel) + '</td>' +
            '<td>' + (routedModel ? '<span style="color:var(--warning)">' + escHtml(routedModel) + '</span>' : '-') + '</td>' +
            '<td>' + ((isDegraded || modelChanged) ? '<span style="color:var(--warning)" title="fallback/routed">' + escHtml(entry.model) + '</span>' : escHtml(entry.model)) + '</td>' +
            '<td><span class="badge-endpoint">' + escHtml(entry.endpoint) + '</span></td>' +
            '<td>' + entry.tokens.toLocaleString() + '</td>' +
            '<td><span class="badge ' + badge + '">' + escHtml(statusLabel) + '</span></td>' +
            '<td><button class="request-detail-btn" onclick="showRequestDetail(' + i + ')" title="' + escHtml(t('stats.requestDetails') || 'Request Details') + '">i</button></td>' +
        '</tr>';
    }
    if (log.length === 0) {
        tableHTML += '<tr><td colspan="11" style="text-align:center;color:var(--text-tertiary);padding:24px">' + t('stats.noRecords') + '</td></tr>';
    }
    tableHTML += '</tbody></table></div>';

    return { activeModels: activeModels, hasData: hasData, tableHTML: tableHTML };
}

function requestStatusLabel(entry) {
    if (entry.status === 'rejected') return t('stats.statusRejected') || 'REJECTED';
    if (entry.status === 'cancelled') return t('stats.statusCancelled') || 'CANCELLED';
    if (entry.status === 'partial' || entry.partial_output) return t('stats.statusPartial') || 'PARTIAL';
    if (entry.status === 'degraded') return t('stats.statusDegraded') || 'DEGRADED';
    if (entry.success === false || entry.status === 'fail') return t('stats.statusFail') || 'FAIL';
    return t('stats.statusOk') || 'OK';
}

function requestKindLabel(entry) {
    return entry && entry.request_kind === 'image_generation'
        ? (t('stats.imageGeneration') || 'Image Generation')
        : (t('stats.textGeneration') || 'Text Generation');
}

function requestKindClass(entry) {
    return entry && entry.request_kind === 'image_generation' ? 'badge-image' : 'badge-endpoint';
}

function requestStatusClass(entry) {
    if (entry.status === 'rejected') return 'badge-rejected';
    if (entry.status === 'cancelled') return 'badge-cancelled';
    if (entry.status === 'partial' || entry.partial_output) return 'badge-partial';
    if (entry.status === 'degraded') return 'badge-degraded';
    if (entry.success === false || entry.status === 'fail') return 'badge-fail';
    return 'badge-ok';
}

function requestDetailValue(value) {
    if (value === true) return 'true';
    if (value === false) return 'false';
    if (value === null || value === '') return '-';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
}

function detailPick(primary, fallback) {
    return primary === null || primary === '' ? fallback : primary;
}

function detailRow(label, value) {
    return '<div class="detail-kv"><div class="detail-label">' + escHtml(label) + '</div><div class="detail-value">' + escHtml(requestDetailValue(value)) + '</div></div>';
}

function detailSection(title, rows) {
    return '<div class="detail-section"><h3>' + escHtml(title) + '</h3><div class="detail-grid">' + rows.join('') + '</div></div>';
}

function showRequestDetail(index) {
    var entry = (window._requestLogDetails || [])[index];
    if (!entry) return;
    var details = entry.details || {};
    var modalContent = document.querySelector('.modal-content');
    if (modalContent) modalContent.classList.add('modal-wide');

    var basicRows = [
        detailRow(t('stats.status') || 'Status', requestStatusLabel(entry)),
        detailRow(t('stats.time') || 'Time', entry.time),
        detailRow(t('stats.fullTime') || 'Full Time', entry.full_time),
        detailRow(t('stats.client') || 'Client', entry.username),
        detailRow(t('stats.key') || 'Key', entry.api_key),
        detailRow(t('stats.endpoint') || 'Client Endpoint', entry.endpoint),
        detailRow(t('stats.requestType') || 'Type', requestKindLabel(entry)),
        detailRow(t('stats.upstreamEndpoint') || 'Upstream Endpoint', entry.upstream_endpoint || details.upstream_endpoint),
        detailRow(t('stats.responsesMode') || 'Responses Mode', entry.responses_mode || details.responses_mode),
        detailRow(t('stats.tokens') || 'Tokens', entry.tokens)
    ];
    var routeRows = [
        detailRow(t('stats.requestedModel') || 'Requested', entry.requested_model || entry.model),
        detailRow(t('stats.routedModel') || 'Routed Target', entry.routed_model || details.routed_model),
        detailRow(t('stats.model') || 'Actual Model', entry.model),
        detailRow(t('stats.provider') || 'Provider', entry.provider),
        detailRow(t('stats.routingMatched') || 'Routing Matched', detailPick(entry.routing_matched, details.routing_matched)),
        detailRow(t('stats.routingRule') || 'Routing Rule', entry.routing_rule_name || details.routing_rule_name || entry.routing_rule_id || details.routing_rule_id),
        detailRow(t('stats.routingReason') || 'Routing Reason', entry.routing_reason || details.routing_reason),
        detailRow(t('stats.attemptedModel') || 'Attempted Model', entry.attempted_model || details.attempted_model),
        detailRow(t('stats.attemptedProvider') || 'Attempted Provider', entry.attempted_provider || details.attempted_provider),
        detailRow(t('stats.stream') || 'Stream', detailPick(entry.stream, details.stream)),
        detailRow(t('stats.partialOutput') || 'Partial Output', detailPick(entry.partial_output, details.partial_output)),
        detailRow(t('stats.fallbackStatus') || 'Fallback Status', entry.fallback_status || details.fallback_status),
        detailRow(t('stats.fallbackReason') || 'Fallback Reason', entry.fallback_reason || details.fallback_reason),
        detailRow(t('stats.nativeAttempted') || 'Native Responses Attempted', detailPick(entry.native_attempted, details.native_attempted)),
        detailRow(t('stats.nativeFailureEndpoint') || 'Native Failure Endpoint', entry.native_failure_endpoint || details.native_failure_endpoint),
        detailRow(t('stats.nativeFailureStatus') || 'Native Failure Status', entry.native_failure_status || details.native_failure_status),
        detailRow(t('stats.nativeFailureReason') || 'Native Failure Reason', entry.native_failure_reason || details.native_failure_reason),
        detailRow(t('stats.nativeFailureMessage') || 'Native Failure Message', entry.native_failure_message || details.native_failure_message),
        detailRow(t('stats.responsesStateful') || 'Stateful Responses Session', detailPick(entry.responses_stateful, details.responses_stateful)),
        detailRow(t('stats.responsesStateMarkers') || 'State Markers', (entry.responses_state_markers || details.responses_state_markers || []).join ? (entry.responses_state_markers || details.responses_state_markers || []).join(', ') : ''),
        detailRow(t('stats.fallbackSafetyDecision') || 'Fallback Safety Decision', entry.fallback_safety_decision || details.fallback_safety_decision),
        detailRow(t('stats.statefulFallbackBlocked') || 'Cross-provider fallback blocked', detailPick(entry.stateful_fallback_blocked, details.stateful_fallback_blocked)),
        detailRow(t('stats.errorTrigger') || 'Error Trigger', entry.error_trigger || details.error_trigger),
        detailRow(t('stats.errorStage') || 'Error Stage', entry.error_stage || details.error_stage)
    ];
    var fallbackAttempts = entry.fallback_attempts || details.fallback_attempts;
    if (Array.isArray(fallbackAttempts) && fallbackAttempts.length) {
        var fbRows = fallbackAttempts.map(function(att) {
            return fallbackAttemptRow(att);
        });
        routeRows.push(detailSection(t('stats.fallbackAttempts') || 'Fallback Chain', fbRows));
    }
    var errorRows = [
        detailRow(t('stats.errorMessage') || 'Error Message', entry.error_message || details.error_message)
    ];
    var imageRows = [];
    if (entry.request_kind === 'image_generation' || details.request_kind === 'image_generation') {
        imageRows = [
            detailRow(t('stats.imageModel') || 'Image Model', entry.image_model || details.image_model),
            detailRow(t('stats.imageBackendType') || 'Image Backend Type', entry.image_backend_type || details.image_backend_type),
            detailRow(t('stats.imageBackendProvider') || 'Image Backend Provider', entry.image_backend_provider || details.image_backend_provider),
            detailRow(t('stats.imageBackendModel') || 'Image Backend Model', entry.image_backend_model || details.image_backend_model),
            detailRow(t('stats.imageFallback') || 'Image Fallback', entry.image_fallback_status || details.image_fallback_status || 'unused'),
            detailRow(t('stats.plannerFallback') || 'Planner Fallback', entry.planner_fallback_status || details.planner_fallback_status),
            detailRow(t('stats.imageCount') || 'Image Count', detailPick(entry.image_count, details.image_count)),
            detailRow(t('stats.imageBytes') || 'Image Bytes', detailPick(entry.image_bytes, details.image_bytes)),
            detailRow(t('stats.imageArtifactCount') || 'Stored Artifacts', detailPick(entry.image_artifact_count, details.image_artifact_count))
        ];
    }

    document.getElementById('modalContent').innerHTML = '<h3>' + escHtml(t('stats.requestDetails') || 'Request Details') + '</h3><div class="request-detail-view">' +
        '<div class="detail-status-line"><span class="badge ' + requestStatusClass(entry) + '">' + escHtml(requestStatusLabel(entry)) + '</span><span class="mono">' + escHtml(entry.endpoint || '') + '</span></div>' +
        detailSection(t('stats.basicInfo') || 'Basic', basicRows) +
        (imageRows.length ? detailSection(t('stats.imageInfo') || 'Image Generation', imageRows) : '') +
        detailSection(t('stats.routingInfo') || 'Routing / Fallback', routeRows) +
        detailSection(t('stats.errorInfo') || 'Error', errorRows) +
        '</div>';
    document.getElementById('modal').style.display = 'flex';
}

function renderStats(stats, createCharts) {
    var summaryHTML = '<div class="dashboard-toolbar">' +
        '<span>' + t('stats.reset') + ': ' + (stats.last_reset || '-') + '  |  ' + t('stats.autoRefresh') + '</span>' +
        '<button class="btn btn-danger btn-sm" onclick="confirmResetStats()">' + t('stats.resetBtn') + '</button>' +
        '</div>' +
        '<div class="summary-cards">' +
        '<div class="summary-card card-purple"><div class="card-icon">&#9636;</div><div class="card-value">' + stats.total_calls.toLocaleString() + '</div><div class="card-label">' + t('stats.totalCalls') + '</div></div>' +
        '<div class="summary-card card-green"><div class="card-icon">&#10003;</div><div class="card-value">' + (stats.health_rate != null ? stats.health_rate : stats.success_rate) + '%</div><div class="card-label">' + t('stats.healthRate') + '</div></div>' +
        '<div class="summary-card card-green"><div class="card-icon">&#10003;</div><div class="card-value">' + stats.success_rate + '%</div><div class="card-label">' + t('stats.successRate') + '</div></div>' +
        '<div class="summary-card card-amber"><div class="card-icon">&#9888;</div><div class="card-value">' + (stats.degraded_calls || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.degradedCalls') + '</div></div>' +
        '<div class="summary-card card-orange"><div class="card-icon">&#9940;</div><div class="card-value">' + (stats.rejected_calls || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.rejectedCalls') + '</div></div>' +
        '<div class="summary-card card-slate"><div class="card-icon">&#10006;</div><div class="card-value">' + (stats.cancelled_calls || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.cancelledCalls') + '</div></div>' +
        '<div class="summary-card card-orange"><div class="card-icon">&#9888;</div><div class="card-value">' + (stats.stateful_fallback_blocked_calls || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.statefulFallbackBlockedCalls') + '</div></div>' +
        '<div class="summary-card card-red"><div class="card-icon">&#9888;</div><div class="card-value">' + stats.failed_calls.toLocaleString() + '</div><div class="card-label">' + t('stats.failedCalls') + '</div></div>' +
        '<div class="summary-card card-purple"><div class="card-icon">&#127912;</div><div class="card-value">' + (stats.image_generation_calls || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.imageGenerationCalls') + '</div></div>' +
        '<div class="summary-card card-blue"><div class="card-icon">&#128444;</div><div class="card-value">' + (stats.image_generation_images || 0).toLocaleString() + '</div><div class="card-label">' + t('stats.generatedImages') + '</div></div>' +
        '<div class="summary-card card-blue"><div class="card-icon">&#9881;</div><div class="card-value">' + Object.keys(stats.stats_by_model || {}).length + '</div><div class="card-label">' + t('stats.activeModels') + '</div></div>' +
        '</div>';

    var rt = _buildRealtimePanel(stats);
    var hasData = rt.hasData;

    // Today's date for default history range
    var today = new Date();
    var todayStr = today.toISOString().split('T')[0];
    var monthAgo = new Date(today.getTime() - 30 * 86400000);
    var monthAgoStr = monthAgo.toISOString().split('T')[0];

    if (createCharts) {
        _destroyCharts();
        _destroyHistoryCharts();
        var dist = stats.distribution || {};
        var tl = stats.timeline || {};

        var html = '<div class="dashboard">' + summaryHTML +
            '<div class="stats-tabs">' +
            '<button id="statsTabRealtime" class="stats-tab active" onclick="switchStatsTab(\'realtime\')">' + t('stats.tabRealtime') + '</button>' +
            '<button id="statsTabHistory" class="stats-tab" onclick="switchStatsTab(\'history\')">' + t('stats.tabHistory') + '</button>' +
            '</div>' +
            '<div id="statsRealtimePanel">';

        if (!hasData) {
            html += '<div class="empty-state"><div class="empty-icon">&#128202;</div><p>' + t('stats.noData') + '</p><p class="empty-sub">' + t('stats.noDataHint') + '</p></div>';
        } else {
            html += '<div class="charts-row">';
            if (dist.labels && dist.labels.length > 0) {
                html += '<div class="chart-card glass"><h3>' + t('stats.modelDist') + '</h3><div class="chart-wrap"><canvas id="chartPie"></canvas></div></div>';
            }
            if (tl.labels && tl.labels.length > 0) {
                html += '<div class="chart-card glass"><div class="trend-header"><h3>' + t('stats.trendChart') + '</h3><div class="trend-toggle"><button class="trend-btn active" id="realtimeTrendCalls" onclick="switchRealtimeTrend(\'calls\')">' + t('stats.trendCalls') + '</button><button class="trend-btn" id="realtimeTrendTokens" onclick="switchRealtimeTrend(\'tokens\')">' + t('stats.trendTokens') + '</button></div></div><div class="chart-wrap"><canvas id="chartLine"></canvas></div></div>';
            }
            html += '</div>' + rt.tableHTML;
        }
        html += '</div>'; // end realtimePanel

        // History panel
        html += '<div id="statsHistoryPanel" style="display:none">' +
            '<div class="history-toolbar">' +
            '<label>' + t('stats.historyFrom') + ' <input type="date" id="historyFrom" value="' + todayStr + '"></label>' +
            '<label>' + t('stats.historyTo') + ' <input type="date" id="historyTo" value="' + todayStr + '"></label>' +
            '<label>' + t('stats.granularity') + ' <select id="historyGranularity">' +
            '<option value="hour" selected>' + t('stats.granHour') + '</option>' +
            '<option value="day">' + t('stats.granDay') + '</option>' +
            '<option value="week">' + t('stats.granWeek') + '</option>' +
            '<option value="month">' + t('stats.granMonth') + '</option>' +
            '</select></label>' +
            '<button class="btn btn-primary btn-sm" onclick="loadHistoryStats()">' + t('stats.query') + '</button>' +
            '</div>' +
            '<div id="historyContent"></div>' +
            '</div>'; // end historyPanel

        html += '</div>'; // end dashboard
        document.getElementById('statsContent').innerHTML = html;

        // Bind Enter key for history query controls
        var histFrom = document.getElementById('historyFrom');
        var histTo = document.getElementById('historyTo');
        var histGran = document.getElementById('historyGranularity');
        if (histFrom) histFrom.addEventListener('keydown', _onHistoryKeydown);
        if (histTo) histTo.addEventListener('keydown', _onHistoryKeydown);
        if (histGran) histGran.addEventListener('keydown', _onHistoryKeydown);

        // Restore tab state
        if (_statsTab === 'history') {
            switchStatsTab('history');
        }

        // Create realtime charts
        var chartColors = ['#818cf8','#34d399','#fbbf24','#f472b6','#f87171','#38bdf8','#a78bfa','#fb923c','#6366f1','#4ade80'];
        if (hasData) {
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
                var tlModels = stats.timeline_models || {};
                if (tlModels.models && tlModels.models.length > 0) {
                    window._realtimeTmData = tlModels;
                    var rtChart = _buildOrupdateTrendChart('chartLine', tlModels, 'calls', _statsCharts, null);
                    window._realtimeTrendChart = rtChart;
                } else {
                    // Fallback: no model breakdown, use simple success/fail
                    window._realtimeTmData = null;
                    window._realtimeTrendChart = null;
                    var lineCtx = document.getElementById('chartLine');
                    if (lineCtx) {
                        _statsCharts.push(new Chart(lineCtx, {
                            type: 'bar',
                            data: { labels: tl.labels, datasets: [
                                { label: t('stats.chartSuccess'), data: tl.success, backgroundColor: '#34d399', borderRadius: 4 },
                                { label: t('stats.chartFail'), data: tl.failed, backgroundColor: '#f87171', borderRadius: 4 }
                            ] },
                            options: { responsive: true, maintainAspectRatio: true, scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, beginAtZero: true } } }
                        }));
                    }
                }
            } else {
                window._realtimeTmData = null;
                window._realtimeTrendChart = null;
            }
        }
    } else {
        // Update mode: only update summary + realtime table + chart data in-place
        var toolbarEl = document.querySelector('#statsContent .dashboard-toolbar');
        var cardsEl = document.querySelector('#statsContent .summary-cards');
        var tableEl = document.querySelector('#statsRealtimePanel .table-card');

        var temp = document.createElement('div');
        temp.innerHTML = summaryHTML;
        if (toolbarEl) toolbarEl.replaceWith(temp.querySelector('.dashboard-toolbar'));
        if (cardsEl) cardsEl.replaceWith(temp.querySelector('.summary-cards'));

        if (tableEl && rt.hasData) {
            var newTable = document.createElement('div');
            newTable.innerHTML = rt.tableHTML;
            tableEl.replaceWith(newTable.firstElementChild);
        }

        // Update chart data in-place
        var dist = stats.distribution || {};
        for (var c = 0; c < _statsCharts.length; c++) {
            var chart = _statsCharts[c];
            try {
                if (chart.canvas.id === 'chartPie' && dist.labels) {
                    chart.data.labels = dist.labels;
                    chart.data.datasets[0].data = dist.counts;
                    chart.update('none');
                }
            } catch (e) {}
        }
        // Trend chart with model stacking needs full rebuild on data change
        var tmData = stats.timeline_models || {};
        if (tmData.models && tmData.models.length > 0 && window._realtimeTrendChart) {
            var currentMode = _realtimeTrendMode;
            window._realtimeTmData = tmData;
            var chartIdx = _statsCharts.indexOf(window._realtimeTrendChart);
            _buildOrupdateTrendChart('chartLine', tmData, currentMode, _statsCharts, window._realtimeTrendChart);
            if (chartIdx < 0) _statsCharts.push(window._realtimeTrendChart);
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

    var html = '<div class="model-selector-wrap">';
    html += '<input type="text" class="model-selector-filter" id="' + prefix + '_filter" placeholder="' + escAttr(t('users.filterModels')) + '" oninput="filterModelSelector(\'' + prefix + '\')" autocomplete="off">';
    html += '<div class="model-selector">';
    html += '<div class="model-selector-all"><label class="model-selector-item">' +
        '<input type="checkbox" id="' + prefix + '_all"' + (allChecked ? ' checked' : '') + ' onchange="toggleModelAll(\'' + prefix + '\')">' +
        '<strong>' + t('users.allModels') + '</strong> <span style="color:var(--text-tertiary);margin-left:4px;font-size:11px;">(' + t('users.wildcard') + ')</span>' +
    '</label></div>';

    var pnames = Object.keys(grouped);
    for (var gi = 0; gi < pnames.length; gi++) {
        var pname = pnames[gi];
        var pmodels = grouped[pname];
        html += '<div class="model-selector-group" data-filter-text="' + escAttr(pname) + '"><div class="model-selector-provider">' + escHtml(pname) + '</div>';
        for (var mi = 0; mi < pmodels.length; mi++) {
            var m = pmodels[mi];
            var modelChecked = allChecked || selected.has(m.id);
            var filterText = [pname, m.provider || '', m.name || '', m.id || ''].join(' ');
            html += '<label class="model-selector-item" data-filter-text="' + escAttr(filterText) + '">' +
                '<input type="checkbox" class="' + prefix + '_model" id="' + prefix + '_' + escAttr(m.id) + '" value="' + escAttr(m.id) + '"' + (modelChecked ? ' checked' : '') + (allChecked ? ' disabled' : '') + '>' +
                escHtml(m.name || m.id) +
            '</label>';
        }
        html += '</div>';
    }
    html += '</div></div>';
    return html;
}

function filterModelSelector(prefix) {
    var input = document.getElementById(prefix + '_filter');
    var query = input ? input.value.trim().toLowerCase() : '';
    var groups = document.querySelectorAll('.model-selector-group');
    for (var gi = 0; gi < groups.length; gi++) {
        var group = groups[gi];
        var groupText = (group.getAttribute('data-filter-text') || '').toLowerCase();
        var groupMatches = !query || groupText.indexOf(query) !== -1;
        var visibleCount = 0;
        var items = group.querySelectorAll('.model-selector-item');
        for (var ii = 0; ii < items.length; ii++) {
            var item = items[ii];
            var itemText = (item.getAttribute('data-filter-text') || '').toLowerCase();
            var visible = groupMatches || itemText.indexOf(query) !== -1;
            item.style.display = visible ? '' : 'none';
            if (visible) visibleCount += 1;
        }
        group.style.display = visibleCount ? '' : 'none';
    }
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
        try {
            await navigator.clipboard.writeText(text);
            toast(t('common.copied'), 'success');
            return;
        } catch (e) {
            // HTTP / permission failures still need the textarea fallback.
        }
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
    var modalContent = document.querySelector('.modal-content');
    if (modalContent) modalContent.classList.remove('modal-wide');
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

/* ════════════════════════════════ Request Logs ════════════════════════════════ */

var _logFilterTimer = null;
function onLogFilterInput() {
    if (_logFilterTimer) clearTimeout(_logFilterTimer);
    _logFilterTimer = setTimeout(loadRequestLogs, 350);
}

async function loadRequestLogs() {
    var container = document.getElementById('requestLogsContent');
    if (!container) return;
    container.innerHTML = '<div class="loading-spinner"><span class="spinner"></span><p>' + escHtml(t('stats.loading') || 'Loading...') + '</p></div>';
    try {
        var endpoint = document.getElementById('logEndpointFilter').value;
        var username = document.getElementById('logUsernameFilter').value.trim();
        var status = document.getElementById('logStatusFilter').value;
        var params = [];
        if (endpoint) params.push('endpoint=' + encodeURIComponent(endpoint));
        if (username) params.push('username=' + encodeURIComponent(username));
        if (status) params.push('status=' + encodeURIComponent(status));
        var url = '/admin/request-logs';
        if (params.length) url += '?' + params.join('&');
        var data = await api(url);
        renderRequestLogs(data);
    } catch (e) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠;</div><p>' + escHtml(e.message) + '</p></div>';
        toast(t('stats.loadFail') + ': ' + e.message, 'error');
    }
}

function renderRequestLogs(data) {
    var container = document.getElementById('requestLogsContent');
    if (!container) return;
    var items = (data && data.items) || [];
    var total = (data && data.total) || 0;
    if (!items.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">☰;</div><p>' + escHtml(t('logs.empty') || 'No request logs yet') + '</p></div>';
        return;
    }
    var tableHTML = '<div class="table-wrap"><table class="data-table"><thead><tr>';
    tableHTML += '<th>' + escHtml(t('logs.colTime') || 'Time') + '</th>';
    tableHTML += '<th>' + escHtml(t('stats.requestType') || 'Type') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colEndpoint') || 'Endpoint') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colUser') || 'User') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colModel') || 'Model') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colStatus') || 'Status') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colTokens') || 'Tokens') + '</th>';
    tableHTML += '<th>' + escHtml(t('logs.colActions') || 'Actions') + '</th>';
    tableHTML += '</tr></thead><tbody>';
    items.forEach(function(entry) {
        var ts = (entry.timestamp || '').slice(11) || entry.timestamp || '';
        var badgeClass = entry.status === 'ok' ? 'badge-ok'
            : (entry.status === 'degraded' ? 'badge-degraded'
            : (entry.status === 'rejected' ? 'badge-rejected'
            : (entry.status === 'cancelled' ? 'badge-cancelled'
            : (entry.status === 'partial' ? 'badge-partial' : 'badge-fail'))));
        var statusLabel = entry.status === 'degraded' ? (t('stats.statusDegraded') || 'DEGRADED')
            : (entry.status === 'rejected' ? (t('stats.statusRejected') || 'REJECTED')
            : (entry.status === 'cancelled' ? (t('stats.statusCancelled') || 'CANCELLED')
            : (entry.status === 'partial' ? (t('stats.statusPartial') || 'PARTIAL')
            : (entry.status === 'ok' ? (t('stats.statusOk') || 'OK')
            : (entry.status || '-')))));
        tableHTML += '<tr>';
        tableHTML += '<td class="mono">' + escHtml(ts) + '</td>';
        tableHTML += '<td><span class="badge ' + requestKindClass(entry) + '">' + escHtml(requestKindLabel(entry)) + '</span></td>';
        tableHTML += '<td class="mono">' + escHtml(entry.endpoint || '-') + '</td>';
        tableHTML += '<td>' + escHtml(entry.username || '-') + '</td>';
        tableHTML += '<td class="mono">' + escHtml(entry.model || '-') + '</td>';
        tableHTML += '<td><span class="badge ' + badgeClass + '">' + escHtml(statusLabel) + '</span></td>';
        tableHTML += '<td>' + (entry.tokens || 0) + '</td>';
        tableHTML += '<td><button class="icon-btn" onclick="showRequestLogDetail(' + entry.id + ')" title="' + escHtml(t('logs.viewDetail') || 'View') + '">ℹ</button> ';
        tableHTML += '<button class="icon-btn" onclick="deleteRequestLog(' + entry.id + ')" title="' + escHtml(t('logs.delete') || 'Delete') + '">✖</button></td>';
        tableHTML += '</tr>';
    });
    tableHTML += '</tbody></table></div>';
    var summary = '<div class="history-summary"><span class="history-stat"><strong>' + total + '</strong> ' + escHtml(t('logs.total') || 'total') + '</span></div>';
    container.innerHTML = summary + tableHTML;
}

async function showRequestLogDetail(logId) {
    try {
        var entry = await api('/admin/request-logs/' + logId);
        var content = document.querySelector('.modal-content');
        if (content) content.classList.add('modal-wide');
        var body = '';
        body += '<h3>' + escHtml(t('logs.detailTitle') || 'Request Detail') + ' #' + entry.id + '</h3>';
        var detailBadge = entry.status === 'ok' ? 'badge-ok'
            : (entry.status === 'degraded' ? 'badge-degraded'
            : (entry.status === 'rejected' ? 'badge-rejected'
            : (entry.status === 'cancelled' ? 'badge-cancelled'
            : (entry.status === 'partial' ? 'badge-partial' : 'badge-fail'))));
        body += '<div class="detail-status-line"><span class="badge ' + detailBadge + '">' + escHtml(entry.status || '-') + '</span><span class="mono">' + escHtml(entry.endpoint || '-') + '</span></div>';
        body += '<div class="detail-grid">';
        body += detailRow(t('logs.colTime') || 'Time', entry.timestamp);
        body += detailRow(t('logs.colUser') || 'User', entry.username);
        body += detailRow(t('logs.colEndpoint') || 'Endpoint', entry.endpoint);
        body += detailRow(t('stats.requestType') || 'Type', requestKindLabel(entry));
        body += detailRow(t('stats.requestedModel') || 'Requested', entry.requested_model);
        body += detailRow(t('stats.routedModel') || 'Routed', (entry.details && entry.details.routed_model) || '');
        body += detailRow(t('logs.colModel') || 'Model', entry.model);
        body += detailRow(t('logs.colProvider') || 'Provider', entry.provider);
        body += detailRow(t('logs.colTokens') || 'Tokens', entry.tokens);
        body += detailRow(t('logs.colStatus') || 'Status', entry.status);
        body += detailRow(t('logs.colError') || 'Error', entry.error || '-');
        body += '</div>';
        if (entry.request_kind === 'image_generation') {
            body += '<div class="detail-section"><h3>' + escHtml(t('stats.imageInfo') || 'Image Generation') + '</h3><div class="detail-grid">';
            body += detailRow(t('stats.imageModel') || 'Image Model', entry.image_model || '');
            body += detailRow(t('stats.imageCount') || 'Image Count', entry.image_count || 0);
            body += detailRow(t('stats.imageBytes') || 'Image Bytes', entry.image_bytes || 0);
            body += detailRow(t('stats.imageArtifactCount') || 'Stored Artifacts', entry.image_artifact_count || 0);
            body += detailRow(t('stats.imageRequestedCount') || 'Requested', entry.image_requested_count || 0);
            body += detailRow(t('stats.imageSucceededCount') || 'Succeeded', entry.image_succeeded_count || 0);
            body += detailRow(t('stats.imageFailedCount') || 'Failed', entry.image_failed_count || 0);
            body += detailRow(t('stats.imageRetriedCount') || 'Retried', entry.image_retried_count || 0);
            body += detailRow(t('stats.imageReusedCount') || 'Reused', entry.image_reused_count || 0);
            body += detailRow(t('stats.imageBatchId') || 'Batch ID', entry.image_batch_id || '');
            body += detailRow(t('stats.upstreamEndpoint') || 'Upstream Endpoint', entry.upstream_endpoint || '');
            body += detailRow(t('stats.responsesMode') || 'Responses Mode', entry.responses_mode || '');
            body += '</div></div>';
        }
        body += '<div class="detail-section"><h3>' + escHtml(t('logs.requestBody') || 'Request Body') + '</h3><pre class="json-block">' + escHtml(JSON.stringify(entry.request_body, null, 2)) + '</pre></div>';
        body += '<div class="detail-section"><h3>' + escHtml(t('logs.responseBody') || 'Response Body') + '</h3><pre class="json-block">' + escHtml(JSON.stringify(entry.response_body, null, 2)) + '</pre></div>';
        body += '<div class="detail-section"><h3>' + escHtml(t('logs.details') || 'Routing / Details') + '</h3><pre class="json-block">' + escHtml(JSON.stringify(entry.details, null, 2)) + '</pre></div>';
        document.getElementById('modalContent').innerHTML = body;
        document.getElementById('modal').style.display = 'flex';
    } catch (e) {
        toast(t('logs.loadDetailFail') + ': ' + e.message, 'error');
    }
}

async function deleteRequestLog(logId) {
    if (!confirm(t('logs.confirmDelete') || 'Delete this log entry?')) return;
    try {
        await api('/admin/request-logs/' + logId, { method: 'DELETE' });
        toast(t('logs.deleted') || 'Deleted', 'success');
        loadRequestLogs();
    } catch (e) {
        toast(t('logs.deleteFail') + ': ' + e.message, 'error');
    }
}

async function confirmClearRequestLogs() {
    if (!confirm(t('logs.confirmClear') || 'Clear ALL request logs?')) return;
    try {
        var r = await api('/admin/request-logs/clear', { method: 'POST' });
        toast((t('logs.cleared') || 'Cleared') + ' (' + (r.removed || 0) + ')', 'success');
        loadRequestLogs();
    } catch (e) {
        toast(t('logs.clearFail') + ': ' + e.message, 'error');
    }
}

/* ════════════════════════════════ System Logs ════════════════════════════════ */

var _systemLogFilterTimer = null;
function onSystemLogFilterInput() {
    if (_systemLogFilterTimer) clearTimeout(_systemLogFilterTimer);
    _systemLogFilterTimer = setTimeout(loadSystemLogs, 350);
}

async function loadSystemLogMeta(force) {
    var container = document.getElementById('systemLogsContent');
    if (container) container.innerHTML = '<div class="loading-spinner"><span class="spinner"></span><p>' + escHtml(t('stats.loading') || 'Loading...') + '</p></div>';
    try {
        if (!window._systemLogMeta || force) {
            window._systemLogMeta = await api('/admin/system-logs/meta');
        }
        renderSystemLogFilters(window._systemLogMeta || {});
        await loadSystemLogs();
    } catch (e) {
        if (container) container.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠;</div><p>' + escHtml(e.message) + '</p></div>';
        toast((t('systemLogs.loadFail') || 'Failed to load system logs') + ': ' + e.message, 'error');
    }
}

function renderSystemLogFilters(meta) {
    var dateSel = document.getElementById('sysLogDateFilter');
    var channelSel = document.getElementById('sysLogChannelFilter');
    if (!dateSel || !channelSel) return;
    var oldDate = dateSel.value;
    var oldChannel = channelSel.value || 'app';
    var dates = meta.dates || [];
    var channels = meta.channels || [];
    var dateHTML = '';
    if (!dates.length) {
        dateHTML = '<option value="">' + escHtml(t('systemLogs.noDates') || 'No log dates') + '</option>';
    } else {
        dates.forEach(function(date) {
            dateHTML += '<option value="' + escHtml(date) + '">' + escHtml(date) + '</option>';
        });
    }
    dateSel.innerHTML = dateHTML;
    if (oldDate && dates.indexOf(oldDate) >= 0) dateSel.value = oldDate;

    var channelHTML = '';
    channels.forEach(function(ch) {
        var label = t('systemLogs.channel.' + ch.id) || (ch.id + ' (' + ch.filename + ')');
        channelHTML += '<option value="' + escHtml(ch.id) + '">' + escHtml(label) + '</option>';
    });
    channelSel.innerHTML = channelHTML;
    if (oldChannel) channelSel.value = oldChannel;
}

async function loadSystemLogs() {
    var container = document.getElementById('systemLogsContent');
    if (!container) return;
    var dateSel = document.getElementById('sysLogDateFilter');
    var channelSel = document.getElementById('sysLogChannelFilter');
    var levelSel = document.getElementById('sysLogLevelFilter');
    var searchInput = document.getElementById('sysLogSearchFilter');
    var date = dateSel ? dateSel.value : '';
    var channel = channelSel ? channelSel.value : 'app';
    var level = levelSel ? levelSel.value : '';
    var query = searchInput ? searchInput.value.trim() : '';
    container.innerHTML = '<div class="loading-spinner"><span class="spinner"></span><p>' + escHtml(t('stats.loading') || 'Loading...') + '</p></div>';
    try {
        var params = ['limit=300'];
        if (date) params.push('date=' + encodeURIComponent(date));
        if (channel) params.push('channel=' + encodeURIComponent(channel));
        if (level) params.push('level=' + encodeURIComponent(level));
        if (query) params.push('q=' + encodeURIComponent(query));
        var data = await api('/admin/system-logs?' + params.join('&'));
        renderSystemLogs(data);
    } catch (e) {
        container.innerHTML = '<div class="empty-state"><div class="empty-icon">⚠;</div><p>' + escHtml(e.message) + '</p></div>';
        toast((t('systemLogs.loadFail') || 'Failed to load system logs') + ': ' + e.message, 'error');
    }
}

function renderSystemLogs(data) {
    var container = document.getElementById('systemLogsContent');
    if (!container) return;
    var items = (data && data.items) || [];
    var total = (data && data.total) || 0;
    var summary = '<div class="history-summary"><span class="history-stat"><strong>' + total + '</strong> ' + escHtml(t('systemLogs.total') || 'entries') + '</span>';
    summary += '<span class="history-stat mono">' + escHtml((data.channel || '') + ' / ' + (data.date || '')) + '</span></div>';
    if (!items.length) {
        container.innerHTML = summary + '<div class="empty-state"><div class="empty-icon">☰;</div><p>' + escHtml(t('systemLogs.empty') || 'No system log entries') + '</p></div>';
        return;
    }
    var html = '<div class="system-log-list">';
    items.forEach(function(entry) {
        var level = String(entry.level || '').toUpperCase();
        var badgeClass = level === 'ERROR' ? 'badge-fail' : (level === 'WARNING' ? 'badge-partial' : 'badge-ok');
        html += '<div class="system-log-row">';
        html += '<div class="system-log-meta"><span class="badge ' + badgeClass + '">' + escHtml(level || '-') + '</span>';
        html += '<span class="mono">' + escHtml(entry.ts || '-') + '</span>';
        html += '<span class="mono">#' + escHtml(entry.line || '-') + '</span>';
        if (entry.request_id) html += '<span class="mono">rid=' + escHtml(entry.request_id) + '</span>';
        html += '<span class="mono">' + escHtml(entry.logger || '-') + '</span></div>';
        html += '<pre class="system-log-msg">' + escHtml(entry.msg || entry.raw || '') + '</pre>';
        if (entry.exc) html += '<pre class="system-log-exc">' + escHtml(entry.exc) + '</pre>';
        html += '</div>';
    });
    html += '</div>';
    container.innerHTML = summary + html;
}

/* ════════════════════════════════ Config import/export ════════════════════════════════ */

async function exportConfig() {
    try {
        var include = document.getElementById('exportIncludeSecrets').checked;
        var r = await fetch('/admin/config/export?include_secrets=' + (include ? 'true' : 'false'), {
            headers: { 'Authorization': 'Bearer ' + localStorage.getItem(SESSION_KEY) }
        });
        if (r.status === 401) {
            var errData = await r.json().catch(function() { return {}; });
            if (isSessionAuthError(errData.detail)) {
                handleSessionExpired();
                return;
            }
            throw new Error(errData.detail || 'HTTP ' + r.status);
        }
        if (!r.ok) {
            throw new Error(await r.text());
        }
        var blob = await r.blob();
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        var ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
        a.href = url;
        a.download = 'llm-aio-config-' + ts + '.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        toast(t('config.exported') || 'Exported', 'success');
    } catch (e) {
        toast(t('config.exportFail') + ': ' + e.message, 'error');
    }
}

async function importConfig() {
    var fileInput = document.getElementById('configFileInput');
    var modeSel = document.getElementById('configImportMode');
    var resultEl = document.getElementById('configImportResult');
    if (!fileInput.files || !fileInput.files[0]) {
        toast(t('config.chooseFile') || 'Please select a JSON file first', 'error');
        return;
    }
    try {
        var text = await fileInput.files[0].text();
        var payload = JSON.parse(text);
        payload.mode = modeSel.value || 'skip';
        var r = await api('/admin/config/import', { method: 'POST', body: JSON.stringify(payload) });
        var summary = r.summary || {};
        var lines = [];
        lines.push((t('config.resultProviders') || 'Providers:') + ' ' + JSON.stringify(summary.providers || {}));
        lines.push((t('config.resultRouting') || 'Routing rules:') + ' ' + JSON.stringify(summary.routing_rules || {}));
        lines.push((t('config.resultFallbacks') || 'Fallback policies:') + ' ' + JSON.stringify(summary.fallback_policies || {}));
        if (resultEl) resultEl.innerHTML = '<pre class="json-block">' + escHtml(lines.join('\n')) + '</pre>';
        toast(t('config.imported') || 'Imported', 'success');
    } catch (e) {
        if (resultEl) resultEl.innerHTML = '<div class="error-text">' + escHtml(e.message) + '</div>';
        toast(t('config.importFail') + ': ' + e.message, 'error');
    }
}

async function downloadJson(endpoint, filenamePrefix, successKey, failKey) {
    try {
        var r = await fetch(endpoint, {
            headers: { 'Authorization': 'Bearer ' + localStorage.getItem(SESSION_KEY) }
        });
        if (r.status === 401) {
            var errData = await r.json().catch(function() { return {}; });
            if (isSessionAuthError(errData.detail)) {
                handleSessionExpired();
                return;
            }
            throw new Error(errData.detail || 'HTTP ' + r.status);
        }
        if (!r.ok) throw new Error(await r.text());
        var blob = await r.blob();
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        var ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
        a.href = url;
        a.download = filenamePrefix + '-' + ts + '.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        toast(t(successKey) || 'Exported', 'success');
    } catch (e) {
        toast((t(failKey) || 'Export failed') + ': ' + e.message, 'error');
    }
}

async function exportUsers() {
    await downloadJson('/admin/users/export', 'llm-aio-users', 'config.usersExported', 'config.usersExportFail');
}

async function importUsers() {
    var fileInput = document.getElementById('usersFileInput');
    var modeSel = document.getElementById('usersImportMode');
    var resultEl = document.getElementById('usersImportResult');
    if (!fileInput.files || !fileInput.files[0]) {
        toast(t('config.chooseFile') || 'Please select a JSON file first', 'error');
        return;
    }
    try {
        var text = await fileInput.files[0].text();
        var payload = JSON.parse(text);
        payload.mode = modeSel.value || 'skip';
        var r = await api('/admin/users/import', { method: 'POST', body: JSON.stringify(payload) });
        var summary = r.summary || {};
        var lines = [];
        lines.push((t('config.resultUsers') || 'Users:') + ' ' + JSON.stringify(summary.users || {}));
        lines.push((t('config.resultApiKeys') || 'API keys:') + ' ' + JSON.stringify(summary.api_keys || {}));
        if (resultEl) resultEl.innerHTML = '<pre class="json-block">' + escHtml(lines.join('\n')) + '</pre>';
        toast(t('config.usersImported') || 'Imported', 'success');
        await loadUsers();
    } catch (e) {
        if (resultEl) resultEl.innerHTML = '<div class="error-text">' + escHtml(e.message) + '</div>';
        toast((t('config.usersImportFail') || 'Import failed') + ': ' + e.message, 'error');
    }
}

/* ════════════════════════════════ i18n additions ════════════════════════════════ */

Object.assign(I18N.zh, {
    'nav.requestLogs': '请求日志',
    'nav.systemLogs': '系统日志',
    'nav.config': '配置导入/导出',
    'logs.title': '请求/响应日志',
    'logs.allEndpoints': '全部端点',
    'logs.allStatus': '全部状态',
    'logs.filterUser': '按用户名筛选',
    'logs.refresh': '刷新',
    'logs.clear': '清空',
    'logs.empty': '还没有请求日志',
    'logs.colTime': '时间',
    'logs.colEndpoint': '端点',
    'logs.colUser': '用户',
    'logs.colModel': '模型',
    'logs.colStatus': '状态',
    'logs.colTokens': 'Tokens',
    'logs.colActions': '操作',
    'logs.colProvider': 'Provider',
    'logs.colError': '错误',
    'logs.viewDetail': '查看详情',
    'logs.delete': '删除',
    'logs.detailTitle': '请求详情',
    'logs.requestBody': '请求体',
    'logs.responseBody': '响应体',
    'logs.details': '路由/详情',
    'logs.total': '总记录',
    'logs.confirmDelete': '确认删除该条日志？',
    'logs.deleted': '已删除',
    'logs.deleteFail': '删除失败',
    'logs.confirmClear': '确认清空所有请求日志？',
    'logs.cleared': '已清空',
    'logs.clearFail': '清空失败',
    'logs.loadDetailFail': '加载详情失败',
    'stats.upstreamEndpoint': '提供商端点',
    'stats.requestType': '类型',
    'stats.textGeneration': '文本生成',
    'stats.imageGeneration': '图像生成',
    'stats.imageGenerationCalls': '生图调用',
    'stats.generatedImages': '生成图像数',
    'stats.imageInfo': '图像生成',
    'stats.imageModel': '生图模型',
    'stats.imageCount': '图像数量',
    'stats.imageBytes': '图像字节数',
    'stats.imageArtifactCount': '已存储图像',
    'stats.imageBackendType': '生图后端类型',
    'stats.imageBackendProvider': '生图后端提供商',
    'stats.imageBackendModel': '生图后端模型',
    'stats.imageFallback': '生图回退',
    'stats.plannerFallback': '规划回退',
    'stats.imageRequestedCount': '请求次数',
    'stats.imageSucceededCount': '成功次数',
    'stats.imageFailedCount': '失败次数',
    'stats.imageRetriedCount': '重试次数',
    'stats.imageReusedCount': '复用次数',
    'stats.imageBatchId': '批次 ID',
    'systemLogs.title': '系统日志',
    'systemLogs.allLevels': '全部级别',
    'systemLogs.search': '搜索日志',
    'systemLogs.empty': '没有系统日志记录',
    'systemLogs.noDates': '暂无日志日期',
    'systemLogs.total': '条日志',
    'systemLogs.loadFail': '加载系统日志失败',
    'systemLogs.channel.access': '访问日志',
    'systemLogs.channel.error': '错误日志',
    'systemLogs.channel.app': '应用日志',
    'systemLogs.channel.tool_calls': '工具调用日志',
    'systemLogs.channel.request': '请求调试日志',
    'config.title': '配置导入/导出',
    'config.exportTitle': '导出配置',
    'config.exportHint': '生成 providers / routing rules / fallback policies 的 JSON 备份。',
    'config.includeSecrets': '含 api_key（明文）',
    'config.export': '下载 JSON',
    'config.exported': '已导出',
    'config.exportFail': '导出失败',
    'config.importTitle': '导入配置',
    'config.importHint': '上传之前导出的 JSON 文件，可选择 skip / replace / merge 冲突策略。',
    'config.mode': '冲突策略',
    'config.modeSkip': 'skip (保留现有)',
    'config.modeReplace': 'replace (覆盖)',
    'config.modeMerge': 'merge (仅更新非空字段)',
    'config.import': '导入',
    'config.chooseFile': '请选择 JSON 文件',
    'config.imported': '导入完成',
    'config.importFail': '导入失败',
    'config.resultProviders': 'Providers:',
    'config.resultRouting': 'Routing rules:',
    'config.resultFallbacks': 'Fallback policies:',
    'config.exportUsersTitle': '导出用户',
    'config.exportUsersHint': '单独导出用户和 API Key，方便迁移到另一台机器。',
    'config.exportUsers': '下载用户 JSON',
    'config.importUsersTitle': '导入用户',
    'config.importUsersHint': '上传用户 JSON，可选择 skip / replace / merge 冲突策略。',
    'config.importUsers': '导入用户',
    'config.usersExported': '用户已导出',
    'config.usersExportFail': '用户导出失败',
    'config.usersImported': '用户导入完成',
    'config.usersImportFail': '用户导入失败',
    'config.resultUsers': '用户:',
    'config.resultApiKeys': 'API Keys:'
});

Object.assign(I18N.en, {
    'nav.requestLogs': 'Request Logs',
    'nav.systemLogs': 'System Logs',
    'nav.config': 'Config Import/Export',
    'logs.title': 'Request/Response Logs',
    'logs.allEndpoints': 'All endpoints',
    'logs.allStatus': 'All statuses',
    'logs.filterUser': 'Filter by user',
    'logs.refresh': 'Refresh',
    'logs.clear': 'Clear',
    'logs.empty': 'No request logs yet',
    'logs.colTime': 'Time',
    'logs.colEndpoint': 'Endpoint',
    'logs.colUser': 'User',
    'logs.colModel': 'Model',
    'logs.colStatus': 'Status',
    'logs.colTokens': 'Tokens',
    'logs.colActions': 'Actions',
    'logs.colProvider': 'Provider',
    'logs.colError': 'Error',
    'logs.viewDetail': 'View detail',
    'logs.delete': 'Delete',
    'logs.detailTitle': 'Request Detail',
    'logs.requestBody': 'Request Body',
    'logs.responseBody': 'Response Body',
    'logs.details': 'Routing / Details',
    'logs.total': 'total',
    'logs.confirmDelete': 'Delete this log entry?',
    'logs.deleted': 'Deleted',
    'logs.deleteFail': 'Delete failed',
    'logs.confirmClear': 'Clear ALL request logs?',
    'logs.cleared': 'Cleared',
    'logs.clearFail': 'Clear failed',
    'logs.loadDetailFail': 'Failed to load detail',
    'stats.upstreamEndpoint': 'Upstream Endpoint',
    'stats.requestType': 'Type',
    'stats.textGeneration': 'Text Generation',
    'stats.imageGeneration': 'Image Generation',
    'stats.imageGenerationCalls': 'Image Calls',
    'stats.generatedImages': 'Generated Images',
    'stats.imageInfo': 'Image Generation',
    'stats.imageModel': 'Image Model',
    'stats.imageCount': 'Image Count',
    'stats.imageBytes': 'Image Bytes',
    'stats.imageArtifactCount': 'Stored Images',
    'stats.imageBackendType': 'Image Backend Type',
    'stats.imageBackendProvider': 'Image Backend Provider',
    'stats.imageBackendModel': 'Image Backend Model',
    'stats.imageFallback': 'Image Fallback',
    'stats.plannerFallback': 'Planner Fallback',
    'stats.imageRequestedCount': 'Requested',
    'stats.imageSucceededCount': 'Succeeded',
    'stats.imageFailedCount': 'Failed',
    'stats.imageRetriedCount': 'Retried',
    'stats.imageReusedCount': 'Reused',
    'stats.imageBatchId': 'Batch ID',
    'systemLogs.title': 'System Logs',
    'systemLogs.allLevels': 'All levels',
    'systemLogs.search': 'Search logs',
    'systemLogs.empty': 'No system log entries',
    'systemLogs.noDates': 'No log dates',
    'systemLogs.total': 'entries',
    'systemLogs.loadFail': 'Failed to load system logs',
    'systemLogs.channel.access': 'Access Log',
    'systemLogs.channel.error': 'Error Log',
    'systemLogs.channel.app': 'Application Log',
    'systemLogs.channel.tool_calls': 'Tool Call Log',
    'systemLogs.channel.request': 'Request Debug Log',
    'config.title': 'Config Import/Export',
    'config.exportTitle': 'Export Config',
    'config.exportHint': 'Generate a JSON backup of providers, routing rules, and fallback policies.',
    'config.includeSecrets': 'Include api_key (plaintext)',
    'config.export': 'Download JSON',
    'config.exported': 'Exported',
    'config.exportFail': 'Export failed',
    'config.importTitle': 'Import Config',
    'config.importHint': 'Upload a previously exported JSON file. Choose skip / replace / merge strategy.',
    'config.mode': 'Conflict mode',
    'config.modeSkip': 'skip (keep existing)',
    'config.modeReplace': 'replace (overwrite)',
    'config.modeMerge': 'merge (only update non-empty fields)',
    'config.import': 'Import',
    'config.chooseFile': 'Please select a JSON file',
    'config.imported': 'Imported',
    'config.importFail': 'Import failed',
    'config.resultProviders': 'Providers:',
    'config.resultRouting': 'Routing rules:',
    'config.resultFallbacks': 'Fallback policies:',
    'config.exportUsersTitle': 'Export Users',
    'config.exportUsersHint': 'Export users and API keys separately for migration to another machine.',
    'config.exportUsers': 'Download Users JSON',
    'config.importUsersTitle': 'Import Users',
    'config.importUsersHint': 'Upload a users JSON file. Choose skip / replace / merge strategy.',
    'config.importUsers': 'Import Users',
    'config.usersExported': 'Users exported',
    'config.usersExportFail': 'User export failed',
    'config.usersImported': 'Users imported',
    'config.usersImportFail': 'User import failed',
    'config.resultUsers': 'Users:',
    'config.resultApiKeys': 'API keys:'
});
/* ═══════════════════════════════ Init ═══════════════════════════════ */

document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    applyI18n();
    window.addEventListener(SESSION_EXPIRED_EVENT, function() {
        showAuthView(t('auth.expired'));
    });
    initAuth().catch(function(err) {
        toast(t('auth.initFail') + ': ' + err.message, 'error');
    });
});
function fallbackAttemptRow(att) {
    var statusLabel = att.status === 'success' ? (t('stats.fallbackAttemptSuccess') || 'Success')
        : att.status === 'failed' ? (t('stats.fallbackAttemptFailed') || 'Failed')
        : att.status === 'skipped' ? (att.reason || 'Skipped')
        : (t('stats.fallbackAttemptStarted') || 'Started');
    var modelLabel = att.model || '-';
    var providerLabel = att.provider || '';
    var targetLabel = providerLabel && modelLabel.indexOf(providerLabel + '/') !== 0
        ? providerLabel + '/' + modelLabel
        : modelLabel;
    var parts = [
        '#' + (att.index + 1),
        '[' + statusLabel + ']',
        targetLabel,
    ];
    if (att.trigger) parts.push('(' + att.trigger + ')');
    var summary = parts.join(' ');
    var detail = att.error_message || '';
    if (detail) summary += '\n' + detail;
    return '<div class="detail-kv" style="grid-column:1/-1"><div class="detail-label">' + escHtml(t('stats.fallbackAttempt') || 'Attempt') + '</div><div class="detail-value">' + escHtml(summary) + '</div></div>';
}
