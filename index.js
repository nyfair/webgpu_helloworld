init = document.querySelector("#init");
cpu = document.querySelector("#cpu");
webgpu = document.querySelector("#webgpu");
texta = document.querySelector("#a");
textb = document.querySelector("#b");
label = document.querySelector("#msg");
size = 10000;
replicate = 50000;
data = new Uint32Array(size);
buf = new Uint32Array(size * replicate);

function arr2hex(array) {
  return [...new Uint8Array(array)]
    .map(x => x.toString(16).padStart(2, '0'))
    .join(' ');
}

init.addEventListener("click", () => {
  window.crypto.getRandomValues(data);
  texta.innerText = arr2hex(data.buffer);
});

cpu.addEventListener("click", () => {
  start = performance.now();
  for (i = 0; i < replicate; i++) {
    for (j = 0; j < size; j++) {
      buf[i*size+j] = data[j] ^ 0xffffffff;
    }
  } 
  end = performance.now();
  x = buf.slice(0, size)
  textb.innerText = arr2hex(x.buffer);
  label.innerText = 'CPU takes ' + Math.round(end - start) + 'ms';
});

webgpu.addEventListener("click", async () => {
  start = performance.now();
  gpu = navigator.gpu;
  adapter = await gpu.requestAdapter();
  device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
  gpudata = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  gpubuf = device.createBuffer({
    size: buf.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gpudata, 0, data);
  shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0)
      var<storage, read_write> gpudata: array<u32>;
      @group(0) @binding(1)
      var<storage, read_write> gpubuf: array<u32>;

      @compute @workgroup_size(1)
      fn calc(
        @builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>
      ) {
        for (var i = 0u; i < ${size}u; i++) {
          gpubuf[global_id.x * ${size} + i] = gpudata[i] ^ 0xffffffff;
        }
      }
  `,
  });
  bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      bindGroupLayout
    ],
  });
  bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gpudata } },
      { binding: 1, resource: { buffer: gpubuf } },
    ],
  });
  pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "calc" },
  });
  end = performance.now();
  commandEncoder = device.createCommandEncoder();
  computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  computePass.dispatchWorkgroups(replicate);
  computePass.end();

  storage = device.createBuffer({
    size: buf.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  commandEncoder.copyBufferToBuffer(gpubuf, 0, storage, 0, buf.byteLength);
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await storage.mapAsync(GPUBufferUsage.MAP_READ, 0, buf.byteLength);
  result = storage.getMappedRange(0, buf.byteLength);
  output = new Uint32Array(result);
  x = output.slice(0, size)
  textb.innerText = arr2hex(x.buffer);
  label.innerText = 'GPU takes ' + Math.round(end - start) + 'ms';
});
