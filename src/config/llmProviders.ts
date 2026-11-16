export interface ProviderPreset {
  id: string;
  label: string;
  baseUrl: string;
  modelHint: string;
  keyHint: string;
  editable: boolean;
}

export const LLM_PROVIDERS: ProviderPreset[] = [
  { id: "default", label: "Backend Default (.env)", baseUrl: "", modelHint: "Configured in backend/.env", keyHint: "Read from backend .env", editable: true },
  { id: "deepseek", label: "DeepSeek", baseUrl: "https://api.deepseek.com/v1", modelHint: "deepseek-chat", keyHint: "sk-...", editable: false },
  { id: "openrouter", label: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", modelHint: "google/gemini-2.5-flash", keyHint: "sk-or-...", editable: false },
  { id: "qwen", label: "Tongyi Qwen (DashScope)", baseUrl: "https://dashscope.aliyuncs.com/compatible-mode/v1", modelHint: "qwen-plus", keyHint: "sk-...", editable: false },
  { id: "zhipu", label: "Zhipu GLM", baseUrl: "https://open.bigmodel.cn/api/paas/v4", modelHint: "glm-4-plus", keyHint: "...", editable: false },
  { id: "kimi", label: "Kimi (Moonshot)", baseUrl: "https://api.moonshot.cn/v1", modelHint: "moonshot-v1-8k", keyHint: "sk-...", editable: false },
  { id: "openai", label: "OpenAI", baseUrl: "https://api.openai.com/v1", modelHint: "gpt-4o-mini", keyHint: "sk-...", editable: false },
  { id: "ollama", label: "Local Ollama", baseUrl: "http://localhost:11434/v1", modelHint: "llama3.1", keyHint: "ollama (any)", editable: false },
  { id: "custom", label: "Custom", baseUrl: "", modelHint: "Enter model name", keyHint: "Enter API key", editable: true },
];

export function detectProviderId(baseUrl: string): string {
  if (!baseUrl) return "default";
  return LLM_PROVIDERS.find((provider) => provider.baseUrl === baseUrl)?.id || "custom";
}

