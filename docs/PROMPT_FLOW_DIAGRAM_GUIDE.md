# Prompt Flow Diagram Guide for KAI System
## Detailed Low-Level Execution Steps for app.eraser

### **OVERVIEW**
This guide provides detailed step-by-step instructions for creating a comprehensive prompt flow diagram in app.eraser. It traces a single prompt from user input through complete model inference, showing all subsystems and data transformations at a granular level.

---

## **SECTION 1: HIGH-LEVEL FLOW STRUCTURE**

```
[User Input] 
    ↓
[CLI Entry Point] 
    ↓
[Model Loading & Tokenization]
    ↓
[Model Partitioning]
    ↓
[Chunk Deployment (K8s)]
    ↓
[Gateway Initialization]
    ↓
[Distributed Token Generation Loop]
    ↓
[Output Generation]
    ↓
[User Response]
```

---

## **SECTION 2: DETAILED STEP-BY-STEP FLOW FOR app.eraser**

### **PHASE 1: USER INPUT → INITIALIZATION (Steps 1-8)**

#### **Step 1: User Submits Prompt via CLI**
- **Module**: `kai_cli.py::cmd_run()`
- **Input**: Command-line arguments
  - `--model`: Model name (e.g., "sshleifer/tiny-gpt2")
  - `--prompt`: User text prompt
  - `--max-tokens`: Token generation limit
  - `--quantize` (optional): Quantization method
- **Output**: Parsed arguments passed to loader
- **Data Flow Type**: Configuration/Parameters
- **Visual**: Rectangle box labeled "CLI Entry Point"

#### **Step 2: Load Model Weights from HuggingFace Hub**
- **Module**: `model/hf_loader.py::HFModelLoader.load_model()`
- **Input**: Model identifier string
- **Process**:
  1. Connect to HuggingFace Hub
  2. Download model configuration (config.json)
  3. Download model weights (pytorch_model.bin)
  4. Cache locally for reuse
- **Output**: Loaded PyTorch model in memory
- **Data Flow Type**: Large tensor weights
- **Visual**: Parallelogram or process box "HuggingFace Loader"

#### **Step 3: Validate Model Architecture**
- **Module**: `model/hf_loader.py::HFModelLoader.validate_architecture()`
- **Input**: Loaded model
- **Process**:
  1. Check if model type is supported (Transformer, CNN, etc.)
  2. Validate layer count and dimensions
  3. Check dtype compatibility
- **Output**: Architecture validation status (success/error)
- **Data Flow Type**: Control/Validation
- **Visual**: Diamond "Architecture Valid?" with Yes/No branches

#### **Step 4: Initialize Tokenizer**
- **Module**: `model/hf_loader.py::HFModelLoader.get_tokenizer()`
- **Input**: Model identifier
- **Process**:
  1. Load tokenizer configuration
  2. Load vocabulary (vocab.json, merges.txt)
  3. Configure special tokens (BOS, EOS, PAD)
- **Output**: PreTrainedTokenizer instance
- **Data Flow Type**: Configuration object
- **Visual**: Rectangle "Tokenizer Initialization"

#### **Step 5: Tokenize Input Prompt**
- **Module**: `model/hf_loader.py::HFModelLoader.encode_prompt()`
- **Input**: Prompt string + Tokenizer
- **Process**:
  1. Convert text to tokens: `"Hello world"` → `[token_ids]`
  2. Add special tokens (BOS at start, optional padding)
  3. Create attention mask (1 for real tokens, 0 for padding)
  4. Stack into batch tensor
- **Output**: 
  - Token IDs tensor (shape: [batch_size, seq_length])
  - Attention mask tensor (shape: [batch_size, seq_length])
- **Data Flow Type**: Token tensor
- **Example**: `"Hello"` → `[198, 256, 15]` (tensor)
- **Visual**: Rectangle "Tokenizer" with arrow showing "token_ids, attention_mask"

#### **Step 6: Estimate Model Partitioning Requirements**
- **Module**: `model/resource_detector.py::ResourceDetector.detect_available_resources()`
- **Input**: None
- **Process**:
  1. Scan Kubernetes cluster for available nodes
  2. Query each node for:
     - Total CPU cores
     - Total RAM (GB)
     - GPU VRAM (GB per GPU)
     - GPU type and count
  3. Calculate total cluster capacity
- **Output**: ClusterResourceProfile
  ```python
  {
    "num_nodes": 3,
    "total_cpu": 24,
    "total_ram_gb": 96,
    "total_gpu_vram_gb": 48,
    "available_for_model": 35,  # After overhead
  }
  ```
- **Data Flow Type**: System metrics
- **Visual**: Rectangle "Cluster Resource Detection"

#### **Step 7: Calculate Layer-to-Node Mapping**
- **Module**: `model/layer_chunker.py::LayerChunker.create_chunks()`
- **Input**: 
  - Loaded model with layer info
  - Cluster resource profile
  - num_chunks parameter (from user or auto-detected)
- **Process**:
  1. Get total model size and layer dimensions
  2. Calculate layers per chunk: `total_layers / num_chunks`
  3. Create LayerChunk objects:
     ```python
     LayerChunk(
       id=0,
       start_layer=0,
       end_layer=12,
       layers=[...],
       weight_bytes=5242880,
     )
     ```
  4. Sort chunks by layer order
- **Output**: List of LayerChunk objects (ordered)
- **Data Flow Type**: Chunk metadata
- **Visual**: Rectangle "Layer Chunker" → shows chunk assignments

#### **Step 8: Shard Model Weights to Chunk Files**
- **Module**: `model/chunker.py::chunk_model()`
- **Input**: Loaded model + chunk specifications
- **Process**:
  1. For each LayerChunk:
     - Extract assigned layers: `model.layers[start:end]`
     - Serialize to PyTorch format (CPU tensor)
     - Save to: `./weights/chunk_{id}.pt`
  2. Verify checksum for each chunk
- **Output**: Files saved to disk at `./weights/chunk_0.pt`, `chunk_1.pt`, etc.
- **Data Flow Type**: Serialized weights (file I/O)
- **Visual**: Rectangle "Weight Sharding" with file icon

---

### **PHASE 2: KUBERNETES DEPLOYMENT (Steps 9-14)**

#### **Step 9: Push Chunk Files to Kubernetes ConfigMap/PVC**
- **Module**: `kubernetes/controller.py::KAIController.upload_weights()`
- **Input**: Chunk files from Step 8
- **Process**:
  1. Create Kubernetes PersistentVolumeClaim (PVC) for weights
  2. Copy chunk files to PVC storage
  3. Verify all chunks are accessible
- **Output**: K8s PVC with all chunk weights
- **Data Flow Type**: File upload to distributed storage
- **Visual**: Rectangle "K8s Weight Upload" → Cylinder "K8s PVC Storage"

#### **Step 10: Deploy Chunk Servers to K8s Nodes**
- **Module**: `kubernetes/controller.py::KAIController.deploy_chunks()`
- **Input**: LayerChunk specifications + Deployment templates
- **Process**:
  1. For each chunk i:
     - Generate deployment YAML from template
     - Set environment variables:
       ```yaml
       CHUNK_ID: i
       NUM_CHUNKS: total_chunks
       MODEL_TYPE: "transformer"
       WEIGHTS_DIR: "/mnt/weights"
       PORT: 50051 + i
       ```
     - Apply to Kubernetes: `kubectl apply -f chunk-deployment-{i}.yaml`
  2. Wait for all pods to reach "Ready" status
- **Output**: Kubernetes Deployment objects + Running Pod instances
- **Data Flow Type**: Deployment manifest + container orchestration
- **Visual**: Rectangles for each chunk deployment → Cylinders for pods

#### **Step 11: Chunk Servers Start and Load Weights**
- **Module**: `model/chunk_server.py::ChunkServer.__init__()` (runs in container)
- **Input**: Environment variables + PVC mounted weights
- **Process** (in each pod):
  1. Read environment variables (CHUNK_ID, NUM_CHUNKS, etc.)
  2. Load assigned chunk weights: `torch.load(f"/mnt/weights/chunk_{id}.pt")`
  3. Construct model layers from weights
  4. Move model to device (GPU if available, else CPU)
  5. Set model to eval mode
  6. Initialize gRPC servicer
  7. Bind to port 50051 + CHUNK_ID
  8. Listen for inference requests
- **Output**: Running gRPC server per chunk
- **Data Flow Type**: Weight loading + server startup
- **Visual**: Multiple rectangles "Chunk Server i" with ports 50051, 50052, etc.

#### **Step 12: Service Discovery - Gateway Discovers Chunk Endpoints**
- **Module**: `model/gateway.py::InferenceGateway.__init__()`
- **Input**: Environment variables listing chunk hosts
  ```
  CHUNK_HOSTS="chunk-0:50051,chunk-1:50052,chunk-2:50053"
  NUM_CHUNKS=3
  ```
- **Process**:
  1. Parse CHUNK_HOSTS from environment
  2. For each chunk host:
     - Attempt gRPC connection: `grpc.aio.secure_channel(host, port)`
     - Create stub: `InferenceServiceStub(channel)`
     - Verify chunk is responsive (health check)
     - Store channel + stub in routing table
  3. Build routing policy (deterministic-latency, load-balance, etc.)
- **Output**: Routing table with active gRPC stubs to each chunk
- **Data Flow Type**: Network discovery + RPC channel setup
- **Visual**: Rectangle "Gateway Initialization" → shows connections to 3 chunk servers

#### **Step 13: Gateway Starts HTTP/gRPC Server**
- **Module**: `model/gateway.py::InferenceGateway.start_server()`
- **Input**: Gateway configuration (port 8080, routing policy)
- **Process**:
  1. Create HTTP server listening on port 8080
  2. Register request handlers for:
     - POST `/infer` - Text inference
     - POST `/infer-batch` - Batch inference
     - GET `/health` - Health check
  3. Start listening for client connections
- **Output**: Running Gateway service on port 8080
- **Data Flow Type**: Server startup + port binding
- **Visual**: Rectangle "Gateway Server Running" with port 8080

#### **Step 14: Verify Full Chain Connectivity**
- **Module**: `model/gateway.py::InferenceGateway.health_check()`
- **Input**: Routing table from Step 12
- **Process**:
  1. For each chunk server:
     - Send gRPC Health check request
     - Verify response received
  2. Log connectivity status
- **Output**: Boolean (all connected) or error
- **Data Flow Type**: Health check signal
- **Visual**: Diamond "Chain Connected?" with Yes/No branches

---

### **PHASE 3: DISTRIBUTED INFERENCE INITIALIZATION (Steps 15-18)**

#### **Step 15: Create DistributedGenerator Instance**
- **Module**: `model/generation.py::DistributedGenerator.__init__()`
- **Input**: 
  - List of LayerChunks (from Step 8)
  - Tokenizer (from Step 4)
  - Device setting (CPU or CUDA)
- **Process**:
  1. Store chunks list in order (must be sequential)
  2. Store tokenizer reference
  3. Move all chunks to target device (if local mode)
  4. Set all chunks to eval mode (no gradients)
  5. Store sampling hyperparameters
- **Output**: Generator object ready to generate tokens
- **Data Flow Type**: Object initialization
- **Visual**: Rectangle "DistributedGenerator Init"

#### **Step 16: Prepare KV-Cache for Inference**
- **Module**: `model/kv_cache_optimizer.py::KVCacheOptimizer.init_cache()`
- **Input**: 
  - Batch size (1 for single prompt)
  - Max sequence length
  - Hidden dimension from model config
- **Process**:
  1. Pre-allocate attention key/value caches:
     ```python
     key_cache = torch.zeros(
       num_layers, batch_size, seq_len, hidden_dim
     )
     value_cache = torch.zeros(
       num_layers, batch_size, seq_len, hidden_dim
     )
     ```
  2. Move caches to device
  3. Initialize cache pointers/indices for incremental update
- **Output**: KV-Cache object with empty caches
- **Data Flow Type**: Tensor memory allocation
- **Visual**: Rectangle "KV-Cache Allocation"

#### **Step 17: Enable Speculative Decoding (Optional)**
- **Module**: `model/speculative_decoder.py::SpeculativeDecoder.__init__()`
- **Input**:
  - Main distributed model
  - Draft model (small, single-device)
  - num_speculative_tokens (e.g., 4)
- **Process**:
  1. Load small draft model (e.g., TinyGPT2)
  2. Move draft model to device
  3. Set draft model to eval mode
  4. Create verification mode configuration
- **Output**: SpeculativeDecoder wrapper
- **Data Flow Type**: Model loading + initialization
- **Visual**: Rectangle "Speculative Decoder Init" (optional branch)

#### **Step 18: Start Autoregressive Generation Loop**
- **Module**: `model/generation.py::DistributedGenerator.generate()`
- **Input**:
  - Token IDs from Step 5
  - max_new_tokens
  - Sampling parameters (temperature, top_k, top_p)
- **Process**:
  1. Initialize output token list with input tokens
  2. Set position = current sequence length
  3. Create generation state: `{position, kv_cache, output_tokens}`
- **Output**: Generation loop ready to produce tokens
- **Data Flow Type**: Loop initialization
- **Visual**: Rectangle "Start Generation Loop"

---

### **PHASE 4: AUTOREGRESSIVE TOKEN GENERATION (Steps 19-35)**
*This phase repeats N times where N = max_new_tokens*

#### **Step 19: Extract Current Token Representation**
- **Module**: `model/generation.py::DistributedGenerator.generate()` (loop start)
- **Input**: 
  - Current output_tokens list
  - Position pointer
  - KV-Cache from Step 16
- **Process**:
  1. Get last token ID: `last_token_id = output_tokens[-1]`
  2. Create input tensor: `input_ids = torch.tensor([[last_token_id]])`
     - Shape: [batch_size=1, seq_len=1]
     - Note: Autoregressive → only process LAST token
  3. Create position IDs: `position_ids = torch.tensor([[current_pos]])`
- **Output**: Input tensor [1, 1] with last token ID
- **Data Flow Type**: Token tensor (single token)
- **Visual**: Rectangle "Extract Last Token"

#### **Step 20: Forward Pass Through Chunk 0 (Layer 0-N)**
- **Module**: `model/chunk_server.py::InferenceServicer.infer()` (Chunk 0)
- **Input**:
  - Input hidden state (token embedding) [1, hidden_dim]
  - Position IDs [1, 1]
  - KV-Cache indices
- **Process**:
  1. Embed token ID if needed: `hidden = embedding(input_ids)` [1, 1, hidden_dim]
  2. Process through layers 0-N in chunk:
     ```python
     for layer in chunk_0.layers:
         hidden, kv_pair = layer(hidden, position_ids, kv_cache)
         store_in_kv_cache(kv_pair, layer_idx)
     ```
  3. Output: hidden state after final layer in chunk + KV updates
- **Output**: 
  - Hidden state tensor [1, 1, hidden_dim]
  - Updated KV-Cache entries for layers in this chunk
- **Data Flow Type**: Tensor through neural network layers
- **Visual**: Rectangle "Chunk 0 Forward Pass" with flow showing intermediate tensor

#### **Step 21: Serialize Hidden State for Network Transfer**
- **Module**: `model/gateway.py::_serialize_tensor()`
- **Input**: Hidden state tensor [1, 1, hidden_dim] from Chunk 0
- **Process**:
  1. Use `torch.save()` to serialize: `buffer = io.BytesIO()`
  2. Compress/optimize for network (if applicable)
  3. Calculate size: e.g., 768 floats = ~3KB
- **Output**: Binary buffer (bytes)
- **Data Flow Type**: Serialized tensor
- **Visual**: Rectangle "Serialize Hidden State" with size annotation

#### **Step 22: Send Hidden State via gRPC to Chunk 1**
- **Module**: `model/gateway.py::InferenceGateway.route_inference()`
- **Input**: 
  - Serialized hidden state (bytes)
  - Request metadata
  - Routing policy
- **Process**:
  1. Select chunk 1 based on routing policy:
     - Deterministic: chunk[step % num_chunks]
     - Load-balance: chunk with lowest current load
  2. Create gRPC request:
     ```protobuf
     InferenceRequest {
         hidden_state: <bytes>,
         position: current_pos,
         model_id: "model_xyz"
     }
     ```
  3. Call gRPC stub: `await chunk1_stub.infer(request, timeout=5s)`
  4. Wait for response with timeout
- **Output**: gRPC response from Chunk 1
- **Data Flow Type**: Network RPC call
- **Visual**: Arrow "gRPC Call" → Chunk 1 box

#### **Step 23: Chunk 1 Receives Request (Layers N+1 to 2N)**
- **Module**: `model/chunk_server.py::InferenceServicer.infer()` (Chunk 1)
- **Input**: gRPC InferenceRequest with serialized hidden state
- **Process**:
  1. Deserialize hidden state: `hidden = torch.load(bytes)`
  2. Process through layers N+1 to 2N:
     ```python
     for layer in chunk_1.layers:
         hidden, kv_pair = layer(hidden, position_ids, kv_cache)
     ```
- **Output**: 
  - Hidden state after layer 2N
  - Updated KV entries for chunk 1's layers
- **Data Flow Type**: Tensor computation
- **Visual**: Rectangle "Chunk 1 Forward Pass"

#### **Step 24: Route Output to Chunk 2**
- **Module**: Same as Step 22 (gateway routing)
- **Input**: Hidden state from Chunk 1
- **Process**: Same serialization + gRPC call to Chunk 2
- **Output**: gRPC call arrives at Chunk 2
- **Data Flow Type**: Network RPC
- **Visual**: Arrow "gRPC Call" → Chunk 2 box

#### **Step 25: Chunk 2 Processes Final Layers & Outputs Logits**
- **Module**: `model/chunk_server.py::InferenceServicer.infer()` (Chunk 2)
- **Input**: Hidden state from Chunk 1
- **Process**:
  1. Deserialize hidden state
  2. Process through layers 2N+1 to total_layers:
     ```python
     for layer in chunk_2.layers:
         hidden, kv_pair = layer(hidden, position_ids, kv_cache)
     ```
  3. Apply final layer norm (if applicable)
  4. Project to vocabulary size:
     ```python
     logits = final_linear(hidden)  # [1, 1, vocab_size]
     ```
- **Output**: Logits tensor [1, 1, vocab_size] (~50,000 values for GPT)
- **Data Flow Type**: Tensor with vocabulary-sized output
- **Visual**: Rectangle "Chunk 2 Final Layers + Logits Output"

#### **Step 26: Serialize Logits for Return**
- **Module**: `model/gateway.py::_serialize_tensor()`
- **Input**: Logits tensor [1, 1, vocab_size]
- **Process**:
  1. Use `torch.save()` to serialize
  2. Compress if needed
  3. Size: 50K floats ≈ 200KB
- **Output**: Binary buffer (bytes)
- **Data Flow Type**: Serialized tensor
- **Visual**: Rectangle "Serialize Logits"

#### **Step 27: Gateway Receives Logits from Chunk 2**
- **Module**: `model/gateway.py::InferenceGateway._infer_internal()`
- **Input**: gRPC response with serialized logits
- **Process**:
  1. Receive gRPC response from Chunk 2
  2. Deserialize: `logits = _deserialize_tensor(response.data)`
  3. Shape: [1, 1, vocab_size]
- **Output**: Logits in gateway process memory
- **Data Flow Type**: Tensor in memory
- **Visual**: Rectangle "Gateway Receives Logits"

#### **Step 28: Speculative Decoding - Draft Model Generation (if enabled)**
- **Module**: `model/speculative_decoder.py::SpeculativeDecoder.speculate_tokens()`
- **Input**: Logits from Step 27 + draft model
- **Process**:
  1. Use small draft model to quickly predict 4 candidate tokens:
     ```python
     draft_logits = draft_model(hidden_state)
     draft_tokens = torch.multinomial(softmax(draft_logits), k=4)
     # Result: [token_A, token_B, token_C, token_D]
     ```
  2. Create candidate sequence
- **Output**: Speculative token candidates
- **Data Flow Type**: Tensor (optional path)
- **Visual**: Rectangle "Draft Model Speculation" (dashed border = optional)

#### **Step 29: Sample Next Token from Logits**
- **Module**: `model/generation.py::DistributedGenerator._sample_token()`
- **Input**: Logits [1, 1, vocab_size]
- **Process**:
  1. Apply temperature scaling:
     ```python
     logits_scaled = logits / temperature  # lower temp = more greedy
     ```
  2. Convert to probabilities: `probs = softmax(logits_scaled)`
  3. Apply top-k/top-p filtering (if enabled):
     ```python
     if top_k > 0:
         probs[probs < topk_threshold] = 0
         probs = probs / probs.sum()  # re-normalize
     ```
  4. Sample from distribution:
     ```python
     next_token = torch.multinomial(probs, num_samples=1)  # or argmax if greedy
     ```
  5. If speculative decoding: verify token matches verification threshold
- **Output**: Next token ID (single integer)
- **Example**: Output token ID = 2547 (representing "world")
- **Data Flow Type**: Scalar token ID
- **Visual**: Rectangle "Sampling" with sampling parameters noted

#### **Step 30: Check Stopping Conditions**
- **Module**: `model/generation.py::DistributedGenerator.generate()` (loop condition)
- **Input**: 
  - next_token ID
  - output_tokens list length
  - max_new_tokens limit
- **Process**:
  1. Check if token == EOS_TOKEN_ID (end-of-sequence): `if next_token == tokenizer.eos_token_id`
  2. Check if length >= max_new_tokens: `if len(output_tokens) >= max_new_tokens`
  3. Check for stop strings (if configured)
  4. Evaluate any other stopping criteria
- **Output**: Boolean (should_stop)
- **Data Flow Type**: Control flow
- **Visual**: Diamond "Should Stop?" with branches

#### **Step 31: Append Token to Output Sequence**
- **Module**: `model/generation.py::DistributedGenerator.generate()`
- **Input**:
  - next_token ID
  - output_tokens list
- **Process**:
  1. Append token: `output_tokens.append(next_token)`
  2. Increment position counter: `position += 1`
  3. Example sequence growth:
     ```
     Start: [input_tokens...]
     After step 1: [..., 2547]
     After step 2: [..., 2547, 456]
     ```
- **Output**: Updated output_tokens list
- **Data Flow Type**: List mutation
- **Visual**: Rectangle "Append Token to Sequence"

#### **Step 32: Update KV-Cache Indices**
- **Module**: `model/kv_cache_optimizer.py::KVCacheOptimizer.update_indices()`
- **Input**:
  - Current sequence length (before append)
  - KV-Cache object
- **Process**:
  1. Increment cache write position: `cache_pos += 1`
  2. Prepare cache for next forward pass
  3. Note: In autoregressive generation, new KV values for new token are appended
- **Output**: Updated KV-Cache ready for next iteration
- **Data Flow Type**: Cache index/pointer update
- **Visual**: Rectangle "Update KV-Cache"

#### **Step 33: Decode and Accumulate Output Text**
- **Module**: `model/generation.py::DistributedGenerator.generate()`
- **Input**:
  - Last token ID (or full token sequence)
  - Tokenizer
- **Process**:
  1. Decode token to text: `text = tokenizer.decode([next_token])`
     - Example: 2547 → " world"
  2. Add to output_text: `output_text += text`
  3. If streaming: yield token text immediately
  4. Example:
     ```
     output_text = "Hello"  (before)
     output_text = "Hello world"  (after step 1)
     ```
- **Output**: Accumulated text string
- **Data Flow Type**: String accumulation
- **Visual**: Rectangle "Decode & Accumulate Text"

#### **Step 34: Loop Back to Step 19**
- **Module**: Loop control
- **Input**: Current state (output_tokens, position, kv_cache)
- **Process**:
  1. If should_stop == False (from Step 30)
  2. Go back to Step 19 to process next token
  3. Note: Token count < max_new_tokens
- **Output**: Loop continues
- **Data Flow Type**: Control flow (loop)
- **Visual**: Arrow looping back with "Continue Loop" label

#### **Step 35: Exit Generation Loop**
- **Module**: `model/generation.py::DistributedGenerator.generate()`
- **Input**: should_stop == True (from Step 30)
- **Process**:
  1. Break out of generation loop
  2. Keep final output_tokens and output_text
- **Output**: Final token sequence + final text
- **Data Flow Type**: Control flow
- **Visual**: Diamond/Rectangle "Exit Loop" with arrow to Step 36

---

### **PHASE 5: POST-PROCESSING & RESPONSE (Steps 36-42)**

#### **Step 36: Full Sequence Post-Processing**
- **Module**: `model/generation.py::DistributedGenerator._postprocess()`
- **Input**: 
  - Final output_tokens list
  - Original input_tokens (context)
- **Process**:
  1. Extract only generated tokens (remove input):
     ```python
     generated_tokens = output_tokens[len(input_tokens):]
     ```
  2. Apply any post-filtering:
     - Remove special tokens
     - Handle beam artifacts
     - Trim padding tokens
- **Output**: Clean generated token sequence
- **Data Flow Type**: Token list
- **Visual**: Rectangle "Post-Processing"

#### **Step 37: Decode Final Token Sequence to Text**
- **Module**: `model/generation.py::DistributedGenerator.generate()`
- **Input**: Generated token sequence
- **Process**:
  1. Batch decode all tokens:
     ```python
     final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
     ```
  2. Clean up formatting:
     - Strip leading/trailing whitespace
     - Handle escape characters
  3. Result: `"Once upon a time, there was a small village..."`
- **Output**: Final generated text string
- **Data Flow Type**: Decoded text string
- **Visual**: Rectangle "Decode to Text"

#### **Step 38: Collect Generation Metrics**
- **Module**: `monitoring/metrics.py::GenerationMetrics.collect()`
- **Input**:
  - Generation start/end timestamps
  - Token count
  - Cache hit rates
  - Network latencies per chunk
- **Process**:
  1. Calculate metrics:
     ```python
     total_time = end_time - start_time
     tokens_per_sec = num_tokens / total_time
     time_per_token_ms = total_time / num_tokens * 1000
     ```
  2. Aggregate chunk latencies:
     - Chunk 0 forward pass: 45ms
     - Network serialization: 2ms
     - Chunk 1 forward pass: 42ms
     - Network call: 1ms
     - Chunk 2 forward pass: 50ms
     - Total: ~140ms per token
  3. Log metrics
- **Output**: GenerationMetrics object with timestamps, latencies, throughput
- **Data Flow Type**: Metrics/telemetry
- **Visual**: Rectangle "Collect Metrics"

#### **Step 39: Build HTTP Response**
- **Module**: `model/gateway.py::InferenceGateway._build_response()`
- **Input**:
  - Final text
  - Generation metrics
  - Request metadata
- **Process**:
  1. Construct JSON response:
     ```json
     {
       "text": "Once upon a time, there was...",
       "tokens_generated": 45,
       "total_latency_ms": 6300,
       "tokens_per_second": 7.14,
       "model_id": "sshleifer/tiny-gpt2",
       "request_id": "req_xyz123"
     }
     ```
  2. Add headers (Content-Type, Cache-Control, etc.)
- **Output**: HTTP response object
- **Data Flow Type**: Serialized HTTP response
- **Visual**: Rectangle "Build HTTP Response"

#### **Step 40: Send Response to Client**
- **Module**: `model/gateway.py::InferenceGateway.handle_request()` (HTTP handler)
- **Input**: HTTP response object from Step 39
- **Process**:
  1. Serialize response to JSON
  2. Send over HTTP socket
  3. Close connection gracefully
- **Output**: Response received by client
- **Data Flow Type**: HTTP network transmission
- **Visual**: Arrow "HTTP Response" → Client/Browser box

#### **Step 41: Energy Feedback Signal (Background)**
- **Module**: `monitoring/event_bus.py::EventBus` + `model/energy_feedback_loop.py`
- **Input**: 
  - Generation metrics from Step 38
  - Node energy samples collected during inference
- **Process** (background task, parallel to Steps 36-40):
  1. Collect node power samples:
     - Chunk 0 node: avg 120W during inference
     - Chunk 1 node: avg 95W
     - Chunk 2 node: avg 110W
  2. Compare to baselines from history
  3. Emit signal to DEAS scheduler if trend detected:
     ```python
     if power_trending_up:
         EventBus.publish("ENERGY_TREND_WARNING", {
             "node": "chunk_0_node",
             "trend": "rising",
             "current_w": 120
         })
     ```
  4. DEAS responds with rebalancing recommendation (for next inference)
- **Output**: Energy event published to event bus
- **Data Flow Type**: Pub/sub event
- **Visual**: Dashed rectangle "Energy Feedback Loop" (background process)

#### **Step 42: Client Receives and Processes Response**
- **Module**: `dashboard/app.py` or `kai_cli.py` (client side)
- **Input**: HTTP response from Step 40
- **Process**:
  1. Parse JSON response
  2. Extract text
  3. Display to user:
     ```
     >>> python kai_cli.py run --prompt "Once upon a time"
     
     Generated Text:
     "Once upon a time, there was a small village nestled in the mountains..."
     
     Generation Stats:
     - Tokens: 45
     - Time: 6.3s
     - Speed: 7.14 tok/s
     ```
  4. Optionally display metrics/dashboard
- **Output**: User sees generated text in terminal or dashboard UI
- **Data Flow Type**: User-facing output
- **Visual**: Terminal/Dashboard box "User Output"

---

## **SECTION 3: DATA FLOW TYPES LEGEND FOR DIAGRAM**

When creating the diagram in app.eraser, use different arrow styles/colors for different data types:

| Data Type | Arrow Style | Color | Example |
|-----------|------------|-------|---------|
| **Token/Tensor** | Solid thick arrow | Blue | Hidden states, logits, tensors flowing through chunks |
| **Configuration/Metadata** | Solid thin arrow | Gray | Environment variables, chunk specs |
| **Network RPC** | Dashed arrow | Purple | gRPC calls between gateway and chunks |
| **File I/O** | Dotted arrow | Orange | Weight loading, checkpoint saving |
| **Event/Signal** | Curved arrow | Red | Energy feedback signals, events |
| **Control Flow** | Rectangle with label | Black | Loop conditions, branching |
| **Metrics/Telemetry** | Dashed thin arrow | Green | Performance metrics collected |

---

## **SECTION 4: KEY NODES FOR DIAGRAM**

### **User-Facing Components**
- Terminal/CLI (user input)
- Dashboard UI (web interface)
- Client Application

### **Loading & Initialization**
- HuggingFace Hub (model source)
- HF Loader (model download)
- Tokenizer (text → tokens)
- Resource Detector (cluster scan)
- Layer Chunker (model split)
- Weight Sharding (serialization)

### **Kubernetes Infrastructure**
- K8s API Server (orchestration)
- PVC Storage (weight storage)
- Chunk Pods (running containers)
- Gateway Service (entry point)

### **Chunk Servers**
- Chunk 0 Server (layers 0-N)
- Chunk 1 Server (layers N+1-2N)
- Chunk 2 Server (layers 2N+1-end)
- [Repeat for more chunks if applicable]

### **Generation Pipeline**
- Generator Loop (autoregressive core)
- KV-Cache Manager
- Sampling Engine (softmax, top-k/p, multinomial)
- Speculative Decoder (optional draft model)
- Tokenizer (decode tokens→text)

### **Monitoring & Feedback**
- Metrics Collector
- Energy Monitor
- Event Bus
- DEAS Scheduler (background rebalancing logic)

### **Output**
- Gateway HTTP Response
- Client Terminal/Dashboard
- Energy Feedback Signal (to DEAS for next run)

---

## **SECTION 5: APP.ERASER DIAGRAM CREATION GUIDE**

### **Step-by-Step Instructions for app.eraser**

#### **Phase 1: Create Container Structures**

1. **Create Main Flow Container**
   - Add a large container/swimlane labeled "KAI Inference System"
   - This will hold all phases

2. **Create Sub-containers for Each Phase**
   - Within main: 5 sub-containers for Phases 1-5
   - Phase 1: "User Input → Initialization" (Steps 1-8)
   - Phase 2: "K8s Deployment" (Steps 9-14)
   - Phase 3: "Inference Init" (Steps 15-18)
   - Phase 4: "Autoregressive Generation" (Steps 19-35)
   - Phase 5: "Post-Processing & Response" (Steps 36-42)

#### **Phase 2: Add Nodes for Each Step**

3. **Within Phase 1 Container**
   - Add 8 rectangular nodes (one per step)
   - Step 1: "CLI Entry Point"
   - Step 2: "HF Loader"
   - Step 3: "Architecture Valid?" (diamond shape)
   - Step 4: "Tokenizer Init"
   - Step 5: "Tokenize Prompt"
   - Step 6: "Cluster Resource Detection"
   - Step 7: "Layer Chunker"
   - Step 8: "Weight Sharding"
   - Connect with arrows in sequence

4. **Within Phase 2 Container**
   - Add 6 nodes for K8s deployment
   - Connect to external element: "Kubernetes API"
   - Show parallel deployment paths (if multiple chunks)

5. **Within Phase 3 Container**
   - Add 4 initialization nodes
   - Show decision diamond for "Speculative Decoding?" (optional branch)

6. **Within Phase 4 Container** (Most Complex)
   - Add loop structure showing autoregressive generation
   - Central loop with nodes 19-35
   - Show 3 chunk paths (Chunk 0, 1, 2) with parallel lanes
   - Use loop notation to show repetition N times
   - Show decision diamond at Step 30: "Should Stop?"
   - Highlight Steps 22 & 24 with network/RPC icons
   - Show KV-Cache update step

7. **Within Phase 5 Container**
   - Linear sequence: post-process → decode → metrics → response → output
   - Add background box for Step 41 (Energy Feedback) as dashed box

#### **Phase 3: Add External Systems**

8. **Left side: Data Sources**
   - HuggingFace Hub (cloud icon)
   - Kubernetes Cluster (cluster icon)
   - User Client (terminal/browser icon)

9. **Right side: Monitoring**
   - Energy Monitor (gauge icon)
   - Event Bus (message queue icon)
   - DEAS Scheduler (processor icon)

#### **Phase 4: Add Arrows and Data Flows**

10. **Configure Arrow Properties**
    - Token flows: Blue solid thick arrows
    - RPC calls: Purple dashed arrows
    - File I/O: Orange dotted arrows
    - Control: Black solid arrows
    - Events: Red curved arrows

11. **Add Arrow Labels** (at key points)
    - Step 5→7: "token_ids, attention_mask"
    - Step 20: "hidden_state [1, 1, hidden_dim]"
    - Step 22: "gRPC call with serialized tensor"
    - Step 25: "logits [1, 1, vocab_size]"
    - Step 29: "next_token (int)"

12. **Add Data Transformation Boxes** (optional)
    - Between chunks: "Serialize → Network → Deserialize"
    - At sampling: "Softmax → Top-k Filter → Multinomial"

#### **Phase 5: Add Annotations and Details**

13. **Add Iteration Counter**
    - Label loop: "Repeat N times (N = max_new_tokens)"
    - Add example: "e.g., N=50"

14. **Add Component Descriptions** (text boxes)
    - Near each major section, add small text box with:
      - Purpose of that phase
      - Typical latencies
      - Memory usage
    - Example for Phase 4:
      ```
      Phase 4: Autoregressive Generation
      - Per-token latency: ~140ms
      - Memory: KV-Cache grows with sequence
      - Loops: 1 to 100+ times
      ```

15. **Add Legend** (bottom of diagram)
    - Arrow styles and their meanings
    - Node types and their purposes
    - Color coding for data types

16. **Add Timing/Performance Notes**
    - Along data flows: latency annotations
    - Example: "Serialization: 2ms", "Network: 1ms", "Compute: 45ms"

#### **Phase 6: Final Formatting**

17. **Organize Layout**
    - Arrange phases left-to-right chronologically
    - Align nodes vertically within phases
    - Use grid/snap-to-grid for consistency
    - Leave space for external systems on sides

18. **Add Visual Hierarchy**
    - Use color-coding: Phase 1=light blue, Phase 2=light green, etc.
    - Bold node borders for important decision points
    - Larger text for main phases vs. steps

19. **Add Interactivity** (if app.eraser supports)
    - Clickable nodes → show detailed code snippets
    - Hover → show tooltips with module paths
    - Toggle "show optional" layers (speculative decoding, energy feedback)

20. **Export/Save**
    - Save as PDF for documentation
    - Export as PNG for reports
    - Keep editable version in app.eraser for future updates

---

## **SECTION 6: EXAMPLE NODE SPECIFICATIONS**

Here are exact examples of nodes you'd add in app.eraser:

```yaml
Step 5 - Tokenize Prompt:
  Shape: Rectangle
  Color: Light Blue
  Border: 1pt solid black
  Text: "Step 5: Tokenize Prompt\nModule: hf_loader.py\nInput: 'Hello world'\nOutput: [token_ids, attention_mask]"
  Size: Width 150px, Height 100px

Step 20 - Chunk 0 Forward Pass:
  Shape: Rectangle with shadow (to show computation)
  Color: #E3F2FD (light blue)
  Border: 2pt solid blue
  Text: "Step 20: Chunk 0 Forward\nLayers 0-N\nLatency: ~45ms"
  Size: Width 150px, Height 100px
  
Step 30 - Should Stop?:
  Shape: Diamond
  Color: Yellow
  Border: 2pt solid orange
  Text: "Should Stop?\nEOS reached?\nLen >= max?"
  Size: Width 120px, Height 80px

Phase 4 Loop:
  Shape: Curved container
  Color: Transparent (outline only)
  Border: 2pt dashed purple
  Label: "Autoregressive Loop (Repeat N times)"
  Contains: Steps 19-35
```

---

## **SECTION 7: CHECKPOINT QUESTIONS FOR VERIFICATION**

Once you create the diagram in app.eraser, verify it by checking:

- [ ] All 42 steps are represented?
- [ ] Phases are clearly separated?
- [ ] Token data flows shown in blue?
- [ ] Network RPC calls shown in purple dashed?
- [ ] Autoregressive loop clearly marked?
- [ ] Decision points (Step 3, 30) are diamonds?
- [ ] 3 chunk servers shown as parallel paths?
- [ ] KV-Cache shown being updated in loop?
- [ ] Energy feedback shown as background process?
- [ ] External systems (HF Hub, K8s, Client) on edges?
- [ ] Legend provided?
- [ ] All module paths labeled?
- [ ] Latency annotations added?
- [ ] Iteration count labeled?

---

## **SECTION 8: QUICK REFERENCE - STEP MAPPING**

```
Phase 1: Steps 1-8
  ├─ 1: CLI Entry
  ├─ 2: HF Loader
  ├─ 3: Architecture Validation
  ├─ 4: Tokenizer Init
  ├─ 5: Tokenize Prompt
  ├─ 6: Resource Detection
  ├─ 7: Layer Chunker
  └─ 8: Weight Sharding

Phase 2: Steps 9-14
  ├─ 9: K8s Weight Upload
  ├─ 10: Deploy Chunk Servers
  ├─ 11: Load Chunk Weights
  ├─ 12: Service Discovery
  ├─ 13: Gateway Startup
  └─ 14: Chain Verification

Phase 3: Steps 15-18
  ├─ 15: DistributedGenerator Init
  ├─ 16: KV-Cache Allocation
  ├─ 17: Speculative Decoder (optional)
  └─ 18: Start Generation Loop

Phase 4: Steps 19-35 (Autoregressive Loop, repeats N times)
  ├─ 19: Extract Last Token
  ├─ 20: Chunk 0 Forward Pass
  ├─ 21: Serialize Hidden State
  ├─ 22: gRPC → Chunk 1
  ├─ 23: Chunk 1 Forward Pass
  ├─ 24: gRPC → Chunk 2
  ├─ 25: Chunk 2 Logits Output
  ├─ 26: Serialize Logits
  ├─ 27: Gateway Receives Logits
  ├─ 28: Speculative Verification (optional)
  ├─ 29: Sample Next Token
  ├─ 30: Should Stop? (decision)
  ├─ 31: Append Token
  ├─ 32: Update KV-Cache
  ├─ 33: Decode & Accumulate
  ├─ 34: Loop Back (if !should_stop)
  └─ 35: Exit Loop (if should_stop)

Phase 5: Steps 36-42
  ├─ 36: Post-Processing
  ├─ 37: Decode to Text
  ├─ 38: Collect Metrics
  ├─ 39: Build HTTP Response
  ├─ 40: Send Response to Client
  ├─ 41: Energy Feedback (background)
  └─ 42: Client Receives Output
```

---

## **CONCLUSION**

This guide provides complete step-by-step specifications for creating a comprehensive, low-level prompt flow diagram in app.eraser. Follow the sections above to:

1. Understand the complete execution flow (Sections 1-2)
2. Use proper data flow notation (Section 3)
3. Identify all components (Section 4)
4. Create the diagram systematically (Section 5)
5. Verify completeness (Section 7)

The resulting diagram will show exactly how a prompt flows through the KAI system from user input to final output, with all distributed inference details, network communication, and optional advanced features clearly illustrated.
