import { Eye, EyeOff, Key } from "lucide-react";
import { LLM_PROVIDERS } from "@/config/llmProviders";

interface ProviderSettingsProps {
  open: boolean;
  providerId: string;
  baseUrl: string;
  apiKey: string;
  modelName: string;
  showApiKey: boolean;
  onProviderChange: (value: string) => void;
  onBaseUrlChange: (value: string) => void;
  onApiKeyChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onShowApiKeyChange: (value: boolean) => void;
  onClose: () => void;
}

export default function ProviderSettings(props: ProviderSettingsProps) {
  if (!props.open) return null;
  const provider = LLM_PROVIDERS.find((item) => item.id === props.providerId) || LLM_PROVIDERS[0];
  const backendDefault = provider.id === "default";
  return (
    <div className="bg-paper-50/95 backdrop-blur border-b border-paper-300/60 px-6 py-5 animate-fade-in">
      <div className="max-w-4xl mx-auto">
        <h3 className="text-xs tracking-normal font-semibold text-ink-500 mb-4 flex items-center gap-2">
          <Key className="w-3.5 h-3.5 text-signal-blue" />LLM Endpoint Configuration
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs tracking-normal font-medium text-ink-400 mb-1.5">Provider</label>
            <select value={props.providerId} onChange={(event) => props.onProviderChange(event.target.value)} className="bg-paper-50/90 border border-paper-300/60 text-ink-700 focus:outline-none w-full rounded-lg px-3 py-2 text-xs">
              {LLM_PROVIDERS.map((item) => <option key={item.id} value={item.id} className="bg-paper-100">{item.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs tracking-normal font-medium text-ink-400 mb-1.5">Base URL</label>
            <input type="text" value={backendDefault ? "" : props.baseUrl} onChange={(event) => props.onBaseUrlChange(event.target.value)} placeholder={backendDefault ? "Read from backend/.env" : "https://api.example.com/v1"} disabled={backendDefault || !provider.editable} className="bg-paper-50/90 border border-paper-300/60 text-ink-700 focus:outline-none w-full rounded-lg px-3 py-2 text-xs font-mono disabled:opacity-50" />
          </div>
          <div>
            <label className="block text-xs tracking-normal font-medium text-ink-400 mb-1.5">API Key</label>
            <div className="relative">
              <input type={props.showApiKey ? "text" : "password"} value={backendDefault ? "" : props.apiKey} onChange={(event) => props.onApiKeyChange(event.target.value)} placeholder={provider.keyHint} disabled={backendDefault} className="bg-paper-50/90 border border-paper-300/60 text-ink-700 focus:outline-none w-full rounded-lg px-3 py-2 pr-9 text-xs font-mono disabled:opacity-50" />
              {!backendDefault && <button onClick={() => props.onShowApiKeyChange(!props.showApiKey)} className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-400 hover:text-ink-600" type="button">{props.showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}</button>}
            </div>
          </div>
          <div>
            <label className="block text-xs tracking-normal font-medium text-ink-400 mb-1.5">Model</label>
            <input type="text" value={backendDefault ? "" : props.modelName} onChange={(event) => props.onModelNameChange(event.target.value)} placeholder={provider.modelHint} disabled={backendDefault} className="bg-paper-50/90 border border-paper-300/60 text-ink-700 focus:outline-none w-full rounded-lg px-3 py-2 text-xs font-mono disabled:opacity-50" />
            <div className="flex justify-end mt-1.5"><button onClick={props.onClose} className="bg-ink-800 hover:bg-ink-700 disabled:bg-ink-200 text-paper-100 px-3 py-1 rounded-lg text-xs tracking-normal">Close</button></div>
          </div>
        </div>
      </div>
    </div>
  );
}

