import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useMemo,
} from "react";
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Platform,
  ActivityIndicator,
  TextInput,
  Animated,
  Easing,
  KeyboardAvoidingView,
  Dimensions,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import {
  useModel,
  GEMMA_3N_E2B_IT_INT4,
  GEMMA_4_E2B_IT,
  // checkMultimodalSupport, // FIXME: re-enable with image attachment
  checkBackendSupport,
  type MemoryUsage,
} from "react-native-litert-lm";
// FIXME: re-enable with image attachment
// import { launchImageLibrary } from "react-native-image-picker";



// ─── Theme ───────────────────────────────────────────────────────────────────
const T = {
  bg: "#08080C",
  surface: "#111118",
  card: "#16161F",
  elevated: "#1C1C28",
  accent: "#6366F1", // Indigo
  accentGlow: "#818CF8",
  success: "#34D399",
  warning: "#FBBF24",
  error: "#F87171",
  cyan: "#22D3EE",
  text: "#F1F1F4",
  dim: "#6B7280",
  muted: "#3F3F50",
  border: "#23232F",
};

const MONO = Platform.OS === "ios" ? "Menlo" : "monospace";
const { width: SCREEN_W } = Dimensions.get("window");

// ─── Types ───────────────────────────────────────────────────────────────────
type ChatMsg = { role: "user" | "model"; text: string; ts: number; thinking?: string };

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmtBytes(b: number): string {
  if (b === 0) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(b) / Math.log(1024));
  return `${(b / Math.pow(1024, i)).toFixed(i > 1 ? 1 : 0)} ${u[i]}`;
}

// ─── Model options ───────────────────────────────────────────────────────────
const MODELS = {
  gemma3n: { label: "Gemma 3n E2B", size: "1.3 GB", url: GEMMA_3N_E2B_IT_INT4 },
  gemma4: { label: "Gemma 4 E2B", size: "2.6 GB", url: GEMMA_4_E2B_IT },
} as const;
type ModelKey = keyof typeof MODELS;

// ═══════════════════════════════════════════════════════════════════════════════
// App
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  return (
    <SafeAreaProvider>
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  // ── State ─────────────────────────────────────────────────────────────────
  const [sel, setSel] = useState<ModelKey>("gemma4");
  const [backend, setBackend] = useState<"cpu" | "gpu">("cpu");
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState("");
  const [busy, setBusy] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [liveMemory, setLiveMemory] = useState<MemoryUsage | null>(null);
  const [enableSpeculativeDecoding, setEnableSpeculativeDecoding] = useState(false);
  const [enableTools, setEnableTools] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  // FIXME: re-enable with image attachment
  // const [attachedImage, setAttachedImage] = useState<{ uri: string; name: string } | null>(null);
  const [activeBackend, setActiveBackend] = useState<string>(backend);
  const scrollRef = useRef<ScrollView>(null);

  const config = useMemo(
    () => ({
      backend,
      systemPrompt: "You are a helpful assistant. Keep responses concise.",
      maxContextTokens: 4096,
      maxOutputTokens: 1024,
      autoLoad: false,
      enableMemoryTracking: true,
      maxMemorySnapshots: 100,
      enableSpeculativeDecoding,
      enableThinking,
      tools: enableTools
        ? [
          {
            name: "get_current_weather",
            description: "Get the current weather for a location",
            parametersJson: JSON.stringify({
              type: "object",
              properties: {
                location: { type: "string", description: "The city and state, e.g. San Francisco, CA" },
                unit: { type: "string", enum: ["celsius", "fahrenheit"] }
              },
              required: ["location"]
            })
          }
        ]
        : undefined,
    }),
    [backend, enableSpeculativeDecoding, enableThinking, enableTools],
  );

  const {
    model,
    isReady,
    downloadProgress,
    error,
    load,
    deleteModel,
    memorySummary,
  } = useModel(MODELS[sel].url, config);

  // ── Query actual backend after model loads ────────────────────────────────
  useEffect(() => {
    if (isReady && model) {
      try {
        const actual = model.getActiveBackend();
        setActiveBackend(actual);
      } catch {
        setActiveBackend(backend);
      }
    } else {
      setActiveBackend(backend);
    }
  }, [isReady, model, backend]);

  // ── Scroll to bottom on new messages ──────────────────────────────────────
  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  }, [chat, streaming]);

  // ── Send message ──────────────────────────────────────────────────────────
  const send = useCallback(async () => {
    if (!model || busy) return;
    const msg = input.trim();
    if (!msg) return;

    setInput("");
    setBusy(true);

    setChat((prev) => [...prev, { role: "user", text: msg, ts: Date.now() }]);
    setStreaming("");

    try {
      const parts: any[] = [];

      // FIXME: image attachment disabled — re-enable once native image path resolution is fixed

      if (msg) {
        parts.push({ type: "text", text: msg });
      }

      const hasTools = enableTools && config.tools && config.tools.length > 0;

      if (hasTools) {
        // ── Tools enabled: use streaming but suppress first-pass display ──
        // The blocking path (no callback) crashes because the SDK's parser
        // fails on Gemma 4's sometimes-malformed tool call tokens.
        // Streaming works fine — we just don't show the first-pass text.
        setStreaming("Processing...");

        // Stream silently to let the SDK capture tool calls
        const result = await model.execute(parts, () => {});

        if (result.toolCalls && result.toolCalls.length > 0) {
          const toolNames = result.toolCalls.map((tc: any) => tc.name).join(", ");
          setStreaming("Calling " + toolNames + "...");

          const toolResults = result.toolCalls.map((tc: any) => {
            if (tc.name === "get_current_weather") {
              const args = JSON.parse(tc.argumentsJson);
              return {
                name: tc.name,
                responseJson: JSON.stringify({
                  location: args.location,
                  temperature: 22,
                  unit: args.unit || "celsius",
                  condition: "Sunny",
                }),
              };
            }
            return { name: tc.name, responseJson: '{"error": "Unknown tool"}' };
          });

          // Stream the real response after tool execution
          let toolFull = "";
          const toolResult = await model.sendToolResponse(toolResults, (token: string) => {
            toolFull += token;
            setStreaming(toolFull);
          });

          setChat((prev) => [
            ...prev,
            { role: "model", text: toolResult.text, ts: Date.now(), thinking: toolResult.thinkingText || undefined },
          ]);
        } else {
          // Model answered without calling tools
          setChat((prev) => [
            ...prev,
            { role: "model", text: result.text, ts: Date.now(), thinking: result.thinkingText || undefined },
          ]);
        }
      } else {
        // ── No tools: stream normally ─────────────────────────────────────
        setStreaming(enableThinking ? "Thinking..." : "");

        let full = "";
        let hasStartedResponse = false;
        const result = await model.execute(parts, (token: string) => {
          if (token) {
            if (!hasStartedResponse && enableThinking) {
              hasStartedResponse = true;
              full = ""; // Clear "Thinking..." when actual response starts
            }
            full += token;
            setStreaming(full);
          }
        });

        setChat((prev) => [
          ...prev,
          { role: "model", text: result.text, ts: Date.now(), thinking: result.thinkingText || undefined },
        ]);
      }
      setStreaming("");

      // Refresh memory stats
      try {
        setLiveMemory(model.getMemoryUsage());
      } catch { }
    } catch (e: any) {
      setChat((prev) => [
        ...prev,
        { role: "model", text: `Error: ${e.message}`, ts: Date.now() },
      ]);
      setStreaming("");
    } finally {
      setBusy(false);
    }
  }, [model, input, busy]);

  // ── Stats ─────────────────────────────────────────────────────────────────
  const stats = model && isReady ? model.getStats() : null;

  // ── Download state helpers ────────────────────────────────────────────────
  const isDownloading = downloadProgress > 0 && downloadProgress < 1;
  const isLoading = downloadProgress === 1 && !isReady;
  const canInteract = !isReady && !isDownloading && !isLoading;
  const gpuWarning = useMemo(() => checkBackendSupport("gpu"), []);

  return (
    <SafeAreaView style={s.root}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={0}
      >
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <View style={s.header}>
          <View>
            <Text style={s.brand}>
              react-native-<Text style={{ color: T.accent }}>litert-lm</Text>
            </Text>
            <Text style={s.tagline}>
              On-device AI •{" "}
              {Platform.OS === "ios" ? "Metal" : activeBackend.toUpperCase()}
            </Text>
          </View>
          <TouchableOpacity
            style={s.settingsBtn}
            onPress={() => setShowSettings(!showSettings)}
          >
            <Text style={{ fontSize: 18, color: T.text }}>
              {showSettings ? "✕" : "⚙"}
            </Text>
          </TouchableOpacity>
        </View>

        {/* ── Settings drawer ────────────────────────────────────────────── */}
        {showSettings && (
          <View style={s.drawer}>
            <Text style={s.drawerTitle}>Model</Text>
            <View style={s.pillRow}>
              {(Object.keys(MODELS) as ModelKey[]).map((k) => (
                <TouchableOpacity
                  key={k}
                  disabled={!canInteract}
                  onPress={() => setSel(k)}
                  style={[
                    s.pill,
                    sel === k && s.pillActive,
                    !canInteract && { opacity: 0.5 },
                  ]}
                >
                  <Text style={[s.pillText, sel === k && s.pillTextActive]}>
                    {MODELS[k].label}
                  </Text>
                  <Text style={s.pillSub}>{MODELS[k].size}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={[s.drawerTitle, { marginTop: 14 }]}>Backend</Text>
            <View style={s.pillRow}>
              {(["cpu", "gpu"] as const).map((b) => {
                return (
                  <TouchableOpacity
                    key={b}
                    disabled={!canInteract}
                    onPress={() => setBackend(b)}
                    style={[
                      s.pill,
                      backend === b && s.pillActive,
                      !canInteract && { opacity: 0.4 },
                    ]}
                  >
                    <Text
                      style={[s.pillText, backend === b && s.pillTextActive]}
                    >
                      {b.toUpperCase()}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            {gpuWarning ? (
              <Text style={s.backendWarning}>{gpuWarning}</Text>
            ) : null}

            <Text style={[s.drawerTitle, { marginTop: 14 }]}>Features (v0.12.0)</Text>
            <View style={s.pillRow}>
              <TouchableOpacity
                disabled={!canInteract}
                onPress={() => setEnableSpeculativeDecoding(!enableSpeculativeDecoding)}
                style={[
                  s.pill,
                  enableSpeculativeDecoding && s.pillActive,
                  !canInteract && { opacity: 0.5 },
                ]}
              >
                <Text style={[s.pillText, enableSpeculativeDecoding && s.pillTextActive]}>
                  Speculative
                </Text>
                <Text style={s.pillSub}>Multi-token</Text>
              </TouchableOpacity>

              <TouchableOpacity
                disabled={!canInteract || enableThinking}
                onPress={() => setEnableTools(!enableTools)}
                style={[
                  s.pill,
                  enableTools && s.pillActive,
                  (!canInteract || enableThinking) && { opacity: 0.5 },
                ]}
              >
                <Text style={[s.pillText, enableTools && s.pillTextActive]}>
                  Tools
                </Text>
                <Text style={s.pillSub}>Function calling</Text>
              </TouchableOpacity>

              <TouchableOpacity
                disabled={!canInteract || enableTools}
                onPress={() => setEnableThinking(!enableThinking)}
                style={[
                  s.pill,
                  enableThinking && s.pillActive,
                  (!canInteract || enableTools) && { opacity: 0.5 },
                ]}
              >
                <Text style={[s.pillText, enableThinking && s.pillTextActive]}>
                  Thinking
                </Text>
                <Text style={s.pillSub}>Reasoning</Text>
              </TouchableOpacity>
            </View>

            {memorySummary && memorySummary.snapshotCount > 0 && (
              <>
                <Text style={[s.drawerTitle, { marginTop: 14 }]}>Memory</Text>
                <View style={s.memRow}>
                  <MiniStat
                    label="RSS"
                    value={fmtBytes(memorySummary.currentResidentBytes)}
                  />
                  <MiniStat
                    label="Heap"
                    value={fmtBytes(memorySummary.currentNativeHeapBytes)}
                  />
                  <MiniStat
                    label="Avail"
                    value={
                      liveMemory
                        ? fmtBytes(liveMemory.availableMemoryBytes)
                        : "—"
                    }
                  />
                </View>
                <View style={[s.memRow, { marginTop: 6 }]}>
                  <MiniStat
                    label="Peak RSS"
                    value={fmtBytes(memorySummary.peakResidentBytes)}
                  />
                  <MiniStat
                    label="Peak Heap"
                    value={fmtBytes(memorySummary.peakNativeHeapBytes)}
                  />
                  <MiniStat
                    label="Snapshots"
                    value={`${memorySummary.snapshotCount}`}
                  />
                </View>
              </>
            )}

            {isReady && (
              <TouchableOpacity
                style={s.dangerBtn}
                onPress={async () => {
                  const fn =
                    sel === "gemma4"
                      ? "gemma-4-E2B-it.litertlm"
                      : "gemma-3n-E2B-it-int4.litertlm";
                  try {
                    await deleteModel(fn);
                  } catch { }
                }}
              >
                <Text style={s.dangerText}>Delete Cached Model</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {/* ── Status / Load ──────────────────────────────────────────────── */}
        {!isReady && (
          <View style={s.statusCard}>
            <PulseRing active={isDownloading || isLoading} />
            <View style={{ flex: 1, marginLeft: 16 }}>
              <Text style={s.statusTitle}>
                {isDownloading
                  ? `Downloading ${(downloadProgress * 100).toFixed(0)}%`
                  : isLoading
                    ? "Loading engine…"
                    : "Model not loaded"}
              </Text>
              <Text style={s.statusSub}>
                {MODELS[sel].label} • {MODELS[sel].size} •{" "}
                {backend.toUpperCase()}
              </Text>
              {error && <Text style={s.errorText}>{error}</Text>}
            </View>
            {canInteract && (
              <TouchableOpacity style={s.loadBtn} onPress={load}>
                <Text style={s.loadBtnText}>Load</Text>
              </TouchableOpacity>
            )}
            {(isDownloading || isLoading) && (
              <ActivityIndicator color={T.accent} style={{ marginLeft: 12 }} />
            )}
          </View>
        )}

        {/* ── Metrics bar ────────────────────────────────────────────────── */}
        {isReady && (
          <View style={s.metricsBar}>
            <MetricChip
              label="Speed"
              value={
                stats?.tokensPerSecond
                  ? `${stats.tokensPerSecond.toFixed(1)}`
                  : "—"
              }
              unit="tok/s"
              color={T.success}
            />
            <MetricChip
              label="Latency"
              value={stats?.totalTime ? `${stats.totalTime.toFixed(0)}` : "—"}
              unit="ms"
              color={T.cyan}
            />
            <MetricChip
              label="Tokens"
              value={
                stats?.completionTokens
                  ? `${Math.round(stats.completionTokens)}`
                  : "—"
              }
              unit=""
              color={T.warning}
            />
          </View>
        )}

        {/* ── Chat area ──────────────────────────────────────────────────── */}
        <ScrollView
          ref={scrollRef}
          style={s.chatArea}
          contentContainerStyle={s.chatContent}
          keyboardShouldPersistTaps="handled"
        >
          {!isReady && chat.length === 0 && (
            <View style={s.emptyState}>
              <Text style={s.emptyIcon}>✦</Text>
              <Text style={s.emptyTitle}>LiteRT LM</Text>
              <Text style={s.emptySub}>
                Load a model to start chatting.{"\n"}
                All inference runs on-device.
              </Text>
            </View>
          )}

          {isReady && chat.length === 0 && (
            <View style={s.emptyState}>
              <Text style={s.emptyIcon}>💬</Text>
              <Text style={s.emptyTitle}>Ready to chat</Text>
              <Text style={s.emptySub}>
                {MODELS[sel].label} loaded on {backend.toUpperCase()}.{"\n"}
                Send a message to begin.
              </Text>
              <View style={s.suggestRow}>
                {[
                  "What is React Native?",
                  "Tell me a joke",
                  "Explain quantum computing",
                ].map((q) => (
                  <TouchableOpacity
                    key={q}
                    style={s.suggestChip}
                    onPress={() => {
                      setInput(q);
                    }}
                  >
                    <Text style={s.suggestText}>{q}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          )}

          {chat.map((m, i) => (
            <ChatBubble key={i} msg={m} />
          ))}

          {streaming !== "" && (
            <ChatBubble
              msg={{ role: "model", text: streaming, ts: Date.now() }}
              isStreaming
            />
          )}
        </ScrollView>

        {/* ── Input bar ──────────────────────────────────────────────────── */}
        {isReady && (
          <View style={{ backgroundColor: T.bg }}>
            {/* FIXME: Image attachment disabled — resolveAssetUri crashes on Android
               when processing bundled assets or content:// URIs from image picker.
               Needs investigation: the native resizeImageIfNeeded / LiteRT SDK
               image processing pipeline causes a hard crash (app closes).
               Re-enable once image path resolution is fixed end-to-end. */}

            <View style={s.inputBar}>

              <TextInput
                style={s.input}
                placeholder="Message…"
                placeholderTextColor={T.dim}
                value={input}
                onChangeText={setInput}
                editable={!busy}
                onSubmitEditing={send}
                returnKeyType="send"
                multiline
              />
              {busy ? (
                <TouchableOpacity
                  style={[s.sendBtn, { backgroundColor: T.error }]}
                  onPress={() => {
                    model?.stopGeneration();
                  }}
                >
                  <Text style={s.sendIcon}>■</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity
                  style={[s.sendBtn, !input.trim() && { opacity: 0.4 }]}
                  onPress={send}
                  disabled={!input.trim()}
                >
                  <Text style={s.sendIcon}>↑</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Components
// ═══════════════════════════════════════════════════════════════════════════════

function ChatBubble({
  msg,
  isStreaming,
}: {
  msg: ChatMsg;
  isStreaming?: boolean;
}) {
  const isUser = msg.role === "user";
  return (
    <View style={[s.bubbleRow, isUser && { justifyContent: "flex-end" }]}>
      {!isUser && (
        <View style={s.avatar}>
          <Text style={{ fontSize: 12 }}>✦</Text>
        </View>
      )}
      <View style={[s.bubble, isUser ? s.bubbleUser : s.bubbleModel]}>
        {!!msg.thinking && (
          <View style={{ backgroundColor: "rgba(255,255,255,0.05)", borderRadius: 8, padding: 8, marginBottom: 8 }}>
            <Text style={{ color: T.dim, fontSize: 11, fontWeight: "600", marginBottom: 4 }}>Thinking</Text>
            <Text style={{ color: T.dim, fontSize: 12 }}>{msg.thinking}</Text>
          </View>
        )}
        <Text style={[s.bubbleText, isUser && { color: "#fff" }]}>
          {msg.text}
          {isStreaming && <Text style={s.cursor}>▊</Text>}
        </Text>
      </View>
    </View>
  );
}

function MetricChip({
  icon,
  label,
  value,
  unit,
  color,
}: {
  icon?: string;
  label: string;
  value: string;
  unit: string;
  color: string;
}) {
  return (
    <View style={s.metricChip}>
      {icon ? <Text style={{ fontSize: 14 }}>{icon}</Text> : null}
      <View style={icon ? { marginLeft: 6 } : undefined}>
        <Text style={s.metricLabel}>{label}</Text>
        <Text style={[s.metricValue, { color }]}>
          {value} <Text style={s.metricUnit}>{unit}</Text>
        </Text>
      </View>
    </View>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <View style={s.miniStat}>
      <Text style={s.miniLabel}>{label}</Text>
      <Text style={s.miniValue}>{value}</Text>
    </View>
  );
}

function PulseRing({ active }: { active: boolean }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (active) {
      Animated.loop(
        Animated.timing(anim, {
          toValue: 1,
          duration: 1500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ).start();
    } else {
      anim.setValue(0);
    }
  }, [active]);

  const scale = anim.interpolate({ inputRange: [0, 1], outputRange: [1, 1.4] });
  const opacity = anim.interpolate({
    inputRange: [0, 1],
    outputRange: [0.6, 0],
  });

  return (
    <View
      style={{
        width: 40,
        height: 40,
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {active && (
        <Animated.View
          style={{
            position: "absolute",
            width: 40,
            height: 40,
            borderRadius: 20,
            backgroundColor: T.accent,
            transform: [{ scale }],
            opacity,
          }}
        />
      )}
      <View
        style={{
          width: 24,
          height: 24,
          borderRadius: 12,
          backgroundColor: active ? T.accent : T.muted,
        }}
      />
    </View>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Styles
// ═══════════════════════════════════════════════════════════════════════════════
const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: T.bg },

  // Header
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 12,
  },
  brand: {
    fontSize: 26,
    fontWeight: "900",
    color: T.text,
    letterSpacing: -0.5,
  },
  tagline: { fontSize: 12, color: T.dim, marginTop: 2, fontWeight: "500" },
  settingsBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: T.card,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: T.border,
  },

  // Settings drawer
  drawer: {
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 16,
    backgroundColor: T.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: T.border,
  },
  drawerTitle: {
    fontSize: 11,
    fontWeight: "700",
    color: T.dim,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 8,
  },
  pillRow: { flexDirection: "row", gap: 8 },
  pill: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    alignItems: "center",
  },
  pillActive: {
    borderColor: T.accent,
    backgroundColor: "rgba(99,102,241,0.12)",
  },
  pillText: { fontSize: 13, fontWeight: "700", color: T.dim },
  pillTextActive: { color: T.accentGlow },
  pillSub: { fontSize: 10, color: T.dim, marginTop: 2 },
  backendWarning: {
    fontSize: 11,
    color: "#f5a623",
    marginTop: 6,
    lineHeight: 15,
    fontStyle: "italic",
  },
  dangerBtn: {
    marginTop: 14,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: T.error,
    alignItems: "center",
  },
  dangerText: { color: T.error, fontWeight: "700", fontSize: 13 },
  memRow: { flexDirection: "row", gap: 8 },
  miniStat: {
    flex: 1,
    backgroundColor: T.card,
    borderRadius: 10,
    padding: 10,
    borderWidth: 1,
    borderColor: T.border,
  },
  miniLabel: {
    fontSize: 10,
    color: T.dim,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  miniValue: {
    fontSize: 13,
    color: T.text,
    fontWeight: "700",
    fontFamily: MONO,
    marginTop: 2,
  },

  // Status card
  statusCard: {
    marginHorizontal: 16,
    marginBottom: 12,
    padding: 16,
    backgroundColor: T.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: T.border,
    flexDirection: "row",
    alignItems: "center",
  },
  statusTitle: { fontSize: 15, fontWeight: "700", color: T.text },
  statusSub: { fontSize: 12, color: T.dim, marginTop: 2 },
  errorText: { fontSize: 12, color: T.error, marginTop: 4 },
  loadBtn: {
    backgroundColor: T.accent,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    marginLeft: 12,
  },
  loadBtnText: { color: "#fff", fontWeight: "800", fontSize: 14 },

  // Metrics bar
  metricsBar: {
    flexDirection: "row",
    gap: 8,
    marginHorizontal: 16,
    marginBottom: 8,
  },
  metricChip: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: T.surface,
    borderRadius: 12,
    padding: 10,
    borderWidth: 1,
    borderColor: T.border,
  },
  metricLabel: {
    fontSize: 10,
    color: T.dim,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  metricValue: { fontSize: 15, fontWeight: "800", fontFamily: MONO },
  metricUnit: { fontSize: 10, fontWeight: "500", color: T.dim },

  // Chat
  chatArea: { flex: 1 },
  chatContent: { paddingHorizontal: 16, paddingBottom: 12, flexGrow: 1 },
  emptyState: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 60,
  },
  emptyIcon: { fontSize: 36, marginBottom: 12, color: T.accent },
  emptyTitle: { fontSize: 20, fontWeight: "800", color: T.text },
  emptySub: {
    fontSize: 14,
    color: T.dim,
    textAlign: "center",
    marginTop: 6,
    lineHeight: 20,
  },
  suggestRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 20,
    justifyContent: "center",
  },
  suggestChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: T.card,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: T.border,
  },
  suggestText: { fontSize: 13, color: T.accentGlow, fontWeight: "600" },

  // Bubbles
  bubbleRow: { flexDirection: "row", alignItems: "flex-end", marginBottom: 10 },
  avatar: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: T.card,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
    borderWidth: 1,
    borderColor: T.border,
  },
  bubble: {
    maxWidth: SCREEN_W * 0.75,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 18,
  },
  bubbleUser: { backgroundColor: T.accent, borderBottomRightRadius: 4 },
  bubbleModel: {
    backgroundColor: T.card,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: T.border,
  },
  bubbleText: { fontSize: 15, color: T.text, lineHeight: 21 },
  cursor: { color: T.accentGlow, fontSize: 14 },

  // Input
  inputBar: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: T.bg,
    borderTopWidth: 1,
    borderTopColor: T.border,
  },
  input: {
    flex: 1,
    backgroundColor: T.surface,
    borderRadius: 22,
    paddingHorizontal: 18,
    paddingVertical: 12,
    color: T.text,
    fontSize: 15,
    borderWidth: 1,
    borderColor: T.border,
    maxHeight: 100,
  },
  sendBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: T.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  sendIcon: { color: "#fff", fontSize: 20, fontWeight: "900" },

  // Attachments
  attachmentPreview: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "rgba(99,102,241,0.12)",
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: T.border,
  },
  attachmentPreviewText: {
    color: T.accentGlow,
    fontSize: 13,
    fontWeight: "600",
  },
  removeAttachmentBtn: {
    padding: 4,
  },
  removeAttachmentText: {
    color: T.dim,
    fontSize: 14,
    fontWeight: "bold",
  },
  attachBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: T.card,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: T.border,
  },
  attachIcon: {
    fontSize: 18,
    color: T.text,
  },
});
