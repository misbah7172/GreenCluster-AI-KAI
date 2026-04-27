# KAI Prompt Flow - ASCII Architecture & Visual Reference

## **HIGH-LEVEL SYSTEM ARCHITECTURE**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Terminal CLI (kai_cli.py)  │  Dashboard UI (streamlit)  │  API Client  │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               │ "prompt" + "model" + "max_tokens"
               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MODEL LOADING & PREPARATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│  HuggingFace Hub ──[download]──> HFModelLoader ──[tokenize]──> Tokenizer│
│                                      │                           │       │
│                                      ├─[validate]────────────────┤       │
│                                      │                           │       │
│                    LayerChunker <────┴──────────────────────────┘       │
│                         │                                               │
│                    [split model]                                        │
│                         │                                               │
│                    ResourceDetector                                     │
│                         │                                               │
│                    [scan K8s cluster]                                   │
└──────────────┬──────────────┬──────────────┬──────────────────────────┘
               │              │              │
          chunk_0.pt     chunk_1.pt      chunk_2.pt
               │              │              │
               └──────────────┼──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    KUBERNETES INFRASTRUCTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  K8s API Server                                                         │
│  ├─ Upload weights to PVC                                              │
│  └─ Deploy Chunk Servers & Gateway                                     │
│                                                                          │
│  ┌─────────────────┬─────────────────┬─────────────────┐              │
│  │  Chunk Pod 0    │  Chunk Pod 1    │  Chunk Pod 2    │              │
│  ├─────────────────┼─────────────────┼─────────────────┤              │
│  │ ChunkServer     │ ChunkServer     │ ChunkServer     │              │
│  │ (gRPC :50051)   │ (gRPC :50052)   │ (gRPC :50053)   │              │
│  │                 │                 │                 │              │
│  │ Layers 0-N      │ Layers N+1-2N   │ Layers 2N+1-end │              │
│  │ [Load weights]  │ [Load weights]  │ [Load weights]  │              │
│  └────────┬────────┴────────┬────────┴────────┬────────┘              │
│           │                 │                 │                       │
│  ┌────────▼─────────────────▼─────────────────▼────────┐             │
│  │     Gateway Service (HTTP :8080 + gRPC Client)      │             │
│  │                                                      │             │
│  │  ServiceDiscovery ──────> Routing Policy            │             │
│  │                 ──────> InferenceGateway            │             │
│  └────────────────┬─────────────────────────────────────┘             │
│                   │                                                    │
└───────────────────┼────────────────────────────────────────────────────┘
                    │
                    │ HTTP POST /infer
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               DISTRIBUTED INFERENCE GENERATION ENGINE                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DistributedGenerator.generate(prompt, max_new_tokens=50)      │   │
│  │                                                                 │   │
│  │  ┌────────────────────────────────────────────────────────┐    │   │
│  │  │  AUTOREGRESSIVE GENERATION LOOP (for i in 0..N-1)     │    │   │
│  │  ├────────────────────────────────────────────────────────┤    │   │
│  │  │                                                        │    │   │
│  │  │  Iteration i: Generate token i                        │    │   │
│  │  │  ├─ input: output_tokens[0:seq_len]                  │    │   │
│  │  │  ├─ hidden = embedding(output_tokens[-1])            │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Chunk 0] ──────────────────────────────────────┤    │   │
│  │  │  │  hidden = forward(hidden, layers_0_to_N)         │    │   │
│  │  │  │  latency: ~45ms                                  │    │   │
│  │  │  │  output: hidden [1,1,hidden_dim]                 │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Serialization + Network RPC]                  │    │   │
│  │  │  │  serialize(hidden) → bytes                       │    │   │
│  │  │  │  send via gRPC to Chunk 1                        │    │   │
│  │  │  │  latency: ~3ms                                   │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Chunk 1] ──────────────────────────────────────┤    │   │
│  │  │  │  hidden = deserialize(bytes)                     │    │   │
│  │  │  │  hidden = forward(hidden, layers_N+1_to_2N)      │    │   │
│  │  │  │  latency: ~42ms                                  │    │   │
│  │  │  │  output: hidden [1,1,hidden_dim]                 │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Serialization + Network RPC]                  │    │   │
│  │  │  │  serialize(hidden) → bytes                       │    │   │
│  │  │  │  send via gRPC to Chunk 2                        │    │   │
│  │  │  │  latency: ~2ms                                   │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Chunk 2 - Final Layers] ────────────────────┤    │   │
│  │  │  │  hidden = deserialize(bytes)                     │    │   │
│  │  │  │  hidden = forward(hidden, layers_2N+1_to_end)   │    │   │
│  │  │  │  latency: ~50ms                                  │    │   │
│  │  │  │  logits = output_projection(hidden)              │    │   │
│  │  │  │  output: logits [1,1,vocab_size]                │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Sampling from Logits] ─────────────────────┤    │   │
│  │  │  │  (Optional) speculative_decoder.verify()        │    │   │
│  │  │  │  probs = softmax(logits / temperature)          │    │   │
│  │  │  │  apply top_k / top_p filtering                  │    │   │
│  │  │  │  next_token = multinomial_sample(probs)         │    │   │
│  │  │  │  output: next_token (int)                        │    │   │
│  │  │  │                                                   │    │   │
│  │  │  ├─ [Token Post-Processing]                       │    │   │
│  │  │  │  output_tokens.append(next_token)               │    │   │
│  │  │  │  text += tokenizer.decode([next_token])         │    │   │
│  │  │  │  update_kv_cache()                              │    │   │
│  │  │  │                                                   │    │   │
│  │  │  └─ [Decision: Continue Loop?] ──────────────┐    │   │
│  │  │     if next_token == EOS or len >= max:     │    │   │
│  │  │        BREAK                                  │    │   │
│  │  │     else:                                     │    │   │
│  │  │        CONTINUE (next iteration)              │    │   │
│  │  │                                                    │    │   │
│  │  └────────────────────────────────────────────────────┘    │   │
│  │                                                                 │   │
│  │  POST-LOOP: final_text = accumulated text output             │   │
│  │             = "Once upon a time, there was..."              │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ BACKGROUND: Energy Feedback Loop                              │   │
│  │ ├─ Monitor node power consumption during inference            │   │
│  │ ├─ Detect trending upward (warning signal)                    │   │
│  │ ├─ Emit event to DEAS scheduler                              │   │
│  │ └─ DEAS plans rebalancing for next inference batch           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Response (JSON)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE FORMATTING                             │
├─────────────────────────────────────────────────────────────────────────┤
│  {                                                                       │
│    "text": "Once upon a time, there was...",                            │
│    "tokens_generated": 45,                                             │
│    "total_latency_ms": 6300,                                           │
│    "tokens_per_second": 7.14,                                          │
│    "model_id": "sshleifer/tiny-gpt2",                                  │
│    "request_id": "req_abc123"                                          │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER DISPLAY                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  >>> python kai_cli.py run --prompt "Once upon a time"                 │
│                                                                          │
│  Generated:                                                             │
│  Once upon a time, there was a small village nestled in the mountains. │
│  ...                                                                    │
│                                                                          │
│  Stats: 45 tokens in 6.3s (7.14 tok/s)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **SINGLE INFERENCE ITERATION (ZOOMED IN)**

This shows what happens in ONE step of the autoregressive loop:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        SINGLE INFERENCE ITERATION                         ║
║                    (One of N autoregressive steps)                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Iteration i:
    Input state: output_tokens = [input_tokens] + [generated_so_far]
                 position = len(output_tokens) - 1
                 kv_cache = {...previous_kvs...}

    ┌─────────────────────────────────────────────────────────┐
    │ Step 19: Extract Last Token                             │
    ├─────────────────────────────────────────────────────────┤
    │ last_token_id = output_tokens[-1]        # e.g., 2547   │
    │ input_ids = tensor([[last_token_id]])    # shape [1, 1]  │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 20-21: Chunk 0 Forward + Serialize                │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ INPUT: token_embedding [1, 1, 768]                      │
    │                                                          │
    │ PROCESSING:                                             │
    │ for layer_idx in [0, 1, ..., N-1]:                      │
    │   hidden, (key, value) = layer(hidden, kv_cache)       │
    │   kv_cache[layer_idx] = (key, value)                    │
    │                                                          │
    │ OUTPUT: hidden [1, 1, 768] ✓                            │
    │ LATENCY: ~45ms                                          │
    │ SIZE: 768 floats = 3.1 KB                               │
    │                                                          │
    │ SERIALIZE: bytes = torch.save(hidden)                   │
    │ SIZE AFTER: 3.2 KB + torch overhead                     │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   │ bytes (serialized tensor)
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 22: Network RPC to Chunk 1                         │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ GATEWAY:                                                │
    │ ├─ Create gRPC InferenceRequest                         │
    │ ├─ Set hidden_state = bytes                             │
    │ ├─ Call chunk_1_stub.infer(request, timeout=5s)        │
    │ └─ Wait for response                                    │
    │                                                          │
    │ LATENCY: ~3ms (network + serialization overhead)        │
    │                                                          │
    │ RESPONSE: gRPC InferenceResponse with hidden_state bytes│
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 23-24: Chunk 1 Forward + Serialize                │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ INPUT: deserialized hidden [1, 1, 768]                  │
    │                                                          │
    │ PROCESSING:                                             │
    │ for layer_idx in [N, N+1, ..., 2N-1]:                   │
    │   hidden, (key, value) = layer(hidden, kv_cache)       │
    │   kv_cache[layer_idx] = (key, value)                    │
    │                                                          │
    │ OUTPUT: hidden [1, 1, 768]                              │
    │ LATENCY: ~42ms                                          │
    │                                                          │
    │ SERIALIZE & SEND via gRPC to Chunk 2                    │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 25-27: Chunk 2 Forward + Output Logits            │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ INPUT: deserialized hidden [1, 1, 768]                  │
    │                                                          │
    │ PROCESSING:                                             │
    │ for layer_idx in [2N, 2N+1, ..., total_layers-1]:      │
    │   hidden, (key, value) = layer(hidden, kv_cache)       │
    │   kv_cache[layer_idx] = (key, value)                    │
    │                                                          │
    │ FINAL LAYER: logits = output_projection(hidden)         │
    │                                                          │
    │ OUTPUT: logits [1, 1, 50257]  (vocab_size=50257)       │
    │         bytes serialized = 50257 * 4 bytes = 200KB      │
    │ LATENCY: ~50ms                                          │
    │                                                          │
    │ SERIALIZE & RETURN via gRPC to GATEWAY                  │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 29: Sample Next Token                              │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ INPUT: logits [1, 1, 50257] (in gateway)               │
    │                                                          │
    │ SAMPLING PROCESS:                                       │
    │ 1. Apply temperature:                                   │
    │    logits_scaled = logits / 0.9                         │
    │                                                          │
    │ 2. Convert to probabilities:                            │
    │    probs = softmax(logits_scaled)                       │
    │                                                          │
    │ 3. Apply top-k filtering (k=50):                        │
    │    top_k_indices = argtop(probs, k=50)                  │
    │    probs[other_indices] = 0                             │
    │    probs = probs / probs.sum()  (renormalize)           │
    │                                                          │
    │ 4. Sample from distribution:                            │
    │    next_token = multinomial(probs, num_samples=1)       │
    │    # Result: single integer in range [0, 50256]         │
    │                                                          │
    │ OUTPUT: next_token (e.g., 456)  (single integer)       │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 30: Check Stopping Conditions                      │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ should_stop = False                                     │
    │                                                          │
    │ if next_token == tokenizer.eos_token_id:               │
    │     should_stop = True                                  │
    │ elif len(output_tokens) >= max_new_tokens:             │
    │     should_stop = True                                  │
    │ else:                                                   │
    │     should_stop = False                                 │
    │                                                          │
    │ OUTPUT: should_stop (Boolean)                           │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ├─► [True]  ──→ EXIT LOOP (Step 35)
                   │
                   └─► [False] ──→ CONTINUE (Step 31-34)
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 31-33: Append Token & Accumulate                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ Step 31: output_tokens.append(next_token)              │
    │          # Now: [...previous..., 456]                  │
    │          # Sequence grew by 1                          │
    │                                                          │
    │ Step 32: kv_cache indices updated for next iteration    │
    │          position += 1                                  │
    │                                                          │
    │ Step 33: text += tokenizer.decode([next_token])         │
    │          # Decode 456 -> " world"                      │
    │          # Accumulate: text = "Hello world"             │
    │                                                          │
    │ OUTPUT:                                                 │
    │ - output_tokens: one token longer                       │
    │ - text: one word longer                                │
    │ - kv_cache: cached KVs for all layers updated          │
    └──────────────┬──────────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 34: Loop Back (If should_stop == False)            │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │ JUMP BACK TO STEP 19 WITH:                              │
    │ - output_tokens: [original] + [generated_so_far]       │
    │ - position: incremented by 1                            │
    │ - text: accumulated so far                              │
    │ - kv_cache: updated with new KV pairs                   │
    │                                                          │
    │ NEXT ITERATION WILL:                                    │
    │ - Extract new last token (from this iteration's output) │
    │ - Run through chunks 0→1→2 again                        │
    │ - Sample next token                                     │
    │ - Check stopping                                        │
    │ - Repeat until should_stop or max_tokens reached        │
    │                                                          │
    │ TOTAL TIME FOR THIS ITERATION: ~140ms                   │
    │  (45ms + 3ms + 42ms + 2ms + 50ms sampling overhead)     │
    └──────────────────────────────────────────────────────────┘

Total Time for N iterations (N=50):
    50 iterations × 140ms/iteration = 7000ms = 7 seconds

Throughput: 50 tokens / 7 seconds = 7.14 tokens/second
```

---

## **DATA TRANSFORMATIONS AT EACH STEP**

```
┌────────────────────────────────────────────────────────────────┐
│              Data Shape & Type Transformations                │
└────────────────────────────────────────────────────────────────┘

Step 5: Tokenize Prompt
  Input:  "Hello world"                           (string)
  Output: [198, 256, 15]                          (list of ints)
  Output: tensor([1, 1])                          (attention mask)

Step 19: Extract Last Token
  Input:  [198, 256, 15, 2547, 456]              (list of 5 tokens)
  Output: 456                                     (single int)
  Output: tensor([[456]])                         (shape [1, 1])

Step 20: Chunk 0 Forward Pass
  Input:  hidden tensor [1, 1, 768]              (embedded token)
  Output: hidden tensor [1, 1, 768]              (after layers 0-N)

Step 21: Serialize Hidden
  Input:  tensor [1, 1, 768]                     (PyTorch tensor)
  Output: b'\x80\x05...\x00'                      (binary bytes, ~3.2KB)

Step 22: gRPC Transfer
  Input:  b'\x80\x05...' (bytes)                 (over network)
  Output: InferenceResponse.hidden_state          (gRPC message)

Step 25: Chunk 2 Output Logits
  Input:  hidden tensor [1, 1, 768]              (from Chunk 1)
  Output: logits tensor [1, 1, 50257]            (vocab distribution)

Step 29: Sample Token
  Input:  logits [1, 1, 50257]                   (50,257 probabilities)
          probs [1, 1, 50]  (after top-k)        (top 50 only)
  Output: 456                                     (single integer)

Step 33: Decode Token
  Input:  456                                     (token ID)
  Output: " world"                                (string)

Step 37: Decode Full Sequence
  Input:  [198, 256, 15, 2547, 456, ...]        (all generated tokens)
  Output: "Hello world ..."                       (full text)
```

---

## **LATENCY BREAKDOWN PER ITERATION**

```
Per-Token Generation Time (Single Iteration):

  ├─ Token Embedding:           ~1ms
  │  └─ Input lookup, CPU overhead
  │
  ├─ Chunk 0 Forward Pass:      ~45ms
  │  └─ 12 transformer layers × 2-4ms per layer
  │
  ├─ Serialization (Chunk 0 out):  ~0.5ms
  │  └─ torch.save() overhead
  │
  ├─ Network RPC (→ Chunk 1):   ~2.5ms
  │  └─ gRPC call + network latency (local cluster)
  │
  ├─ Chunk 1 Forward Pass:      ~42ms
  │  └─ 12 transformer layers
  │
  ├─ Network RPC (→ Chunk 2):   ~1.5ms
  │  └─ gRPC call (faster due to smaller payload)
  │
  ├─ Chunk 2 Forward Pass:      ~50ms
  │  └─ 12 transformer layers + final projection
  │
  ├─ Logits Serialization:      ~1ms
  │  └─ Larger payload (~200KB)
  │
  ├─ Sampling (softmax+top-k+multinomial):  ~0.5ms
  │  └─ Fast tensor operations
  │
  ├─ Token Decode & Text Append: ~0.3ms
  │  └─ Tokenizer decode
  │
  └─ TOTAL PER ITERATION:       ~145ms

Throughput: 1 / 0.145s = 6.9 tokens/second ✓

For 50 tokens: 50 × 145ms = 7.25 seconds total ✓
```

---

## **CHUNK COMMUNICATION PROTOCOL**

```
┌──────────────────────────────────────────────────────────┐
│  gRPC Message Protocol Between Gateway ↔ Chunks         │
└──────────────────────────────────────────────────────────┘

REQUEST: InferenceRequest
  ├─ hidden_state: bytes             (serialized tensor)
  ├─ position: int                   (current sequence position)
  ├─ model_id: string                (model identifier)
  ├─ request_id: string              (unique request tracking)
  └─ chunk_id: int                   (which chunk receives this)

RESPONSE: InferenceResponse
  ├─ hidden_state: bytes             (serialized output tensor)
  ├─ position: int                   (echo position)
  ├─ latency_ms: float               (computation time)
  ├─ request_id: string              (echo for correlation)
  └─ status: string                  ("OK" or error message)


Flow for Chunk Chain:
  
  Gateway                Chunk 0              Chunk 1              Chunk 2
     │                     │                   │                    │
     ├────[Request]────→   │                   │                    │
     │   hidden [1,1,768]  │                   │                    │
     │                     │                   │                    │
     │                  [forward 45ms]        │                    │
     │                     │                   │                    │
     │                  [serialize]           │                    │
     │                     │                   │                    │
     │  ←────[Response]─── │                   │                    │
     │   hidden [1,1,768]  │                   │                    │
     │                     │                   │                    │
     ├────[Request]────────────────→         │                    │
     │   hidden [1,1,768]  │                   │                    │
     │                     │            [forward 42ms]            │
     │                     │                   │                    │
     │                     │                [serialize]           │
     │                     │                   │                    │
     │ ←────[Response]───────────────────     │                    │
     │   hidden [1,1,768]  │                   │                    │
     │                     │                   │                    │
     ├────[Request]───────────────────────────→                    │
     │   hidden [1,1,768]  │                   │                    │
     │                     │                   │           [forward 50ms]
     │                     │                   │           [logits output]
     │                     │                   │           [serialize]
     │                     │                   │                    │
     │ ←────[Response]───────────────────────── 
     │   logits [1,1,50257] (200KB)
     │
     ├─ Sampling: softmax → top-k → multinomial
     │  Result: next_token (single int)
     │
     ├─ Decode: token → text
     └─ Return to user in HTTP response
```

---

## **KEY SYSTEM PROPERTIES**

```
┌────────────────────────────────────────────────────────────┐
│          System Characteristics & Constraints             │
└────────────────────────────────────────────────────────────┘

THROUGHPUT:
  ├─ Tokens/Second: 6-8 (single prompt on 3 chunks)
  ├─ Throughput limited by: Chunk forward pass time + network latency
  └─ Can improve by: More chunks (lower work per chunk), GPU acceleration

LATENCY:
  ├─ Per-token latency: ~140-150ms
  ├─ First-token latency: ~200-300ms (includes initialization)
  ├─ Max sequence length: 2048 tokens (transformer context window)
  └─ KV-Cache growth: ~768 bytes per layer per token (~12 layers × 768B)

MEMORY:
  ├─ Model weights: ~3GB total (split across 3 chunks = 1GB each)
  ├─ KV-Cache at 2048 tokens: 12 layers × 768B/token × 2048 = 18.8MB
  ├─ Max concurrent requests: Limited by KV-Cache and chunk memory
  └─ Offloading: Optional CPU/disk via TieredWeightManager

NETWORK:
  ├─ Hidden state per transfer: ~3.2KB (serialized [1,1,768])
  ├─ Logits per transfer: ~200KB (serialized [1,1,50257])
  ├─ RPC overhead: ~3-5ms per call (latency not throughput)
  └─ Scalability: Linear with chunk count (N chunks = N-1 hops per token)

FAULT TOLERANCE:
  ├─ Chunk failure: Request timeout → retry or failover
  ├─ Network failure: gRPC exponential backoff
  ├─ Partial generation: Checkpoint/resume capability
  └─ Energy anomaly: Rebalance next batch via DEAS scheduler

OPTIMIZATION OPPORTUNITIES:
  ├─ Speculative Decoding: Reduce per-token latency via draft model
  ├─ Quantization: Reduce model size & memory footprint
  ├─ Batch Processing: Multiple prompts in parallel
  ├─ Layer Fusion: Combine small layers to reduce chunk hops
  └─ Prefetching: Pre-load weights/cache eagerly
```

---

## **QUICK REFERENCE: MODULE INTERACTIONS**

```
kai_cli.py
  └─→ HFModelLoader (load model weights)
      └─→ Tokenizer (encode prompts)
          └─→ LayerChunker (partition model)
              └─→ ResourceDetector (scan K8s cluster)
                  └─→ chunk_model (save weight files)
                      └─→ [Upload to K8s PVC]
                          └─→ KAIController (deploy chunks)
                              ├─→ InferenceServicer (runs in Chunk Pod)
                              │   └─→ chunk_server.py
                              │
                              ├─→ InferenceGateway (HTTP server)
                              │   ├─→ ServiceDiscovery (find chunks)
                              │   └─→ gRPC routing
                              │
                              └─→ DistributedGenerator (generate tokens)
                                  ├─→ Tokenizer (decode)
                                  ├─→ KVCacheOptimizer (manage cache)
                                  ├─→ SpeculativeDecoder (optional)
                                  └─→ Sampling (softmax, top-k, multinomial)

[BACKGROUND]
  EnergyFeedbackLoop
  ├─→ GPUMonitor (collect power)
  ├─→ EventBus (publish signals)
  └─→ DEASScheduler (plan rebalancing)
```

---

## **CREATING THIS IN APP.ERASER: VISUAL HIERARCHY**

When you build this diagram in app.eraser, organize it with:

```
TOP LEVEL (Canvas):
├─ Title: "KAI Distributed LLM Inference - Prompt Flow"
├─ Legend (bottom): Data types, colors, symbols
│
└─ Five Main Phases (left-to-right):

   1. USER INPUT → INITIALIZATION
      [Boxes arranged vertically: CLI → Loader → Validator → Tokenizer → Chunker]
      ├─ Color: Light Blue (#E3F2FD)
      └─ Time: ~500ms
   
   2. KUBERNETES DEPLOYMENT
      [Boxes: Weight Upload → Pod Launch → Server Ready → Discovery → Gateway Init]
      ├─ Color: Light Green (#E8F5E9)
      ├─ Show K8s API on side
      └─ Time: ~2-5 seconds
   
   3. INFERENCE INITIALIZATION
      [Boxes: Gen Init → Cache Alloc → (Optional: Spec Decoder) → Start Loop]
      ├─ Color: Light Orange (#FFF3E0)
      └─ Time: ~100ms
   
   4. AUTOREGRESSIVE GENERATION LOOP (LARGEST SECTION)
      [Loop container with internal steps 19-35]
      ├─ Color: Light Purple (#F3E5F5)
      ├─ Show 3 parallel chunk lanes
      ├─ Show loop arrow with "Repeat N times (N=1 to 50)"
      ├─ Critical path highlighted (computational steps)
      └─ Time: ~140ms × N iterations
   
   5. POST-PROCESSING & RESPONSE
      [Boxes: Cleanup → Decode → Metrics → Response → Output]
      ├─ Color: Light Red (#FFEBEE)
      ├─ Background dashed box for Energy Feedback
      └─ Time: ~50ms

EXTERNAL SYSTEMS (around edges):
├─ LEFT: HuggingFace Hub, Kubernetes API, Client/User
└─ RIGHT: Monitoring, Event Bus, DEAS Scheduler

TOTAL DIAGRAM SIZE: ~1200px width × 800px height (approximate)
```
