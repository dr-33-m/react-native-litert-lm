# Proguard rules for react-native-litert-lm

# Keep our Nitro JSI wrappers and HybridObjects
-keep class com.margelo.nitro.dev.litert.litertlm.** { *; }

# Keep Google's LiteRT-LM SDK classes and JNI hooks
-keep class com.google.ai.edge.litertlm.** { *; }
-keep class dev.litert.litertlm.** { *; }
