/* PooLabVR browser experiment. WebGL2 only; no CUDA, DX12 or dependencies.
 * A GPU transform-feedback pass retains 32 bytes per particle. Four modes:
 * per-flake wind vs a 64x64x32 cache, two stereo draws vs one instanced draw.
 * The optimized stereo path is WebGL batching, not the DX12 shared-load shader.
 */
(() => {
  'use strict';
  const canvas = document.getElementById('snow');
  const fpsElement = document.getElementById('fps');
  const frameElement = document.getElementById('frameMs');
  const gpuElement = document.getElementById('gpuMs');
  const countElement = document.getElementById('particleHud');
  const status = document.getElementById('renderStatus');
  const select = document.getElementById('population');
  const resolutionSelect = document.getElementById('eyeResolution');
  const resolutionHud = document.getElementById('resolutionHud');
  const benchmarkButton = document.getElementById('benchmark');
  const benchmarkStatus = document.getElementById('testStatus');
  const benchmarkBadge = document.getElementById('testBadge');
  const resultsBody = document.getElementById('results');
  const modeButtons = [...document.querySelectorAll('.mode')];
  const modes = ['baseline', 'stereo', 'cells', 'combined'];
  const labels = { baseline: 'Original', stereo: 'Stéréo groupée', cells: 'Vent localisé', combined: 'Combiné' };
  const GRID = 64, PHASES = 32, TEX_WIDTH = GRID * PHASES;
  const resolution = [canvas.width, canvas.height];
  select.value = matchMedia('(max-width: 760px)').matches ? '200000' : '2000000';
  resolutionSelect.value = '200';

  const gl = canvas.getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false,
    premultipliedAlpha: false, preserveDrawingBuffer: false,
    powerPreference: 'high-performance'
  });
  if (!gl) {
    status.textContent = 'WebGL2 indisponible sur cet appareil';
    benchmarkStatus.textContent = 'Cette expérience nécessite WebGL2. Essaie un navigateur compatible.';
    benchmarkButton.disabled = true;
    modeButtons.forEach(b => { b.disabled = true; });
    return;
  }

  const updateSource = `#version 300 es
    precision highp float; precision highp int;
    layout(location=0) in vec4 aPosition;
    layout(location=1) in vec4 aVelocity;
    uniform float uDt, uTime;
    uniform bool uCellWind;
    uniform highp sampler2D uCells;
    out vec4 nextPosition;
    out vec4 nextVelocity;
    void main() {
      vec4 p = aPosition, v = aVelocity;
      float flutter, swirl;
      if (uCellWind) {
        int cx = clamp(int(floor((p.x+128.0)*0.25)),0,63);
        int cz = clamp(int(floor((p.z+128.0)*0.25)),0,63);
        int phase = clamp(int(floor(p.w*32.0)),0,31);
        vec2 field = texelFetch(uCells,ivec2(cx*32+phase,cz),0).rg;
        flutter = field.x; swirl = field.y;
      } else {
        float phase = p.w*6.2831853;
        flutter = sin(phase+uTime*0.85+p.z*0.035);
        swirl = cos(phase*0.73+uTime*0.58+p.x*0.025);
      }
      v.x += (0.14+flutter*0.22*v.w-v.x*0.20)*uDt;
      v.z += (0.07+swirl*0.19*v.w-v.z*0.20)*uDt;
      float terminal = 0.65+1.15*p.w;
      v.y += (-terminal-v.y)*min(1.0,uDt*2.0);
      p.xyz += v.xyz*uDt;
      if(p.y<0.04) {p.y+=18.0;p.x+=(p.w-0.5)*3.6;p.z+=(v.w-0.75)*3.2;}
      if(p.x>=128.0)p.x-=256.0;if(p.x< -128.0)p.x+=256.0;
      if(p.z>=128.0)p.z-=256.0;if(p.z< -128.0)p.z+=256.0;
      nextPosition=p;nextVelocity=v;
      gl_Position=vec4(0.0);
    }
  `;
  const renderSource = `#version 300 es
    precision highp float;precision highp int;
    layout(location=0) in vec4 aPosition;
    layout(location=1) in vec4 aVelocity;
    uniform bool uBatched;
    uniform int uEye;
    out float vAlpha;
    void main() {
      int eye = uBatched ? gl_InstanceID : uEye;
      float eyeX = (eye==0?-0.032:0.032);
      vec3 p = aPosition.xyz;
      p.x += floor((eyeX-p.x+128.0)/256.0)*256.0;
      p.z += floor((-p.z+128.0)/256.0)*256.0;
      vec3 d = vec3(p.x-eyeX,p.y-1.7,p.z);
      float dd = dot(d,d);
      if ((abs(p.x)<4.2&&abs(p.z)<4.2&&p.y<4.05)||
          abs(d.x)>35.0||abs(d.z)>35.0||dd>1225.0||d.z<=0.05) {
        gl_Position=vec4(2.0,2.0,0.0,1.0);gl_PointSize=1.0;vAlpha=0.0;return;
      }
      vec2 xy = d.xy/d.z;
      if (abs(xy.x)>=1.0||abs(xy.y)>=1.0) {
        gl_Position=vec4(2.0,2.0,0.0,1.0);gl_PointSize=1.0;vAlpha=0.0;return;
      }
      // Each eye owns half of the 400x200 render surface.
      float xx=xy.x*0.5+(eye==0?-0.5:0.5);
      gl_Position=vec4(xx,-xy.y,0.0,1.0);
      gl_PointSize=dd<81.0?2.3:1.6;
      float distance=sqrt(dd);
      vAlpha=clamp((35.0-distance)/19.0,0.15,0.85)*(0.65+0.35*aPosition.w);
    }
  `;
  const renderFragment = `#version 300 es
    precision mediump float;
    in float vAlpha;
    out vec4 color;
    void main() {
      vec2 r=gl_PointCoord*2.0-1.0;
      float fade=clamp(1.7-dot(r,r),0.0,1.0);
      color=vec4(0.84,0.92,1.0,vAlpha*fade);
    }
  `;
  const fullScreenVertex = `#version 300 es
    precision highp float;
    void main(){
      vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
      gl_Position=vec4(p*2.0-1.0,0.0,1.0);
    }
  `;
  const backdropFragment = `#version 300 es
    precision highp float;
    out vec4 color;
    uniform vec2 uResolution;
    void main(){
      vec2 uv=gl_FragCoord.xy/uResolution;
      vec2 eyeUv=vec2(fract(uv.x*2.0),uv.y);
      float horizon=smoothstep(0.36,0.39,eyeUv.y);
      vec3 sky=mix(vec3(0.005,0.024,0.055),vec3(0.055,0.12,0.19),eyeUv.y);
      vec3 ground=vec3(0.038,0.060,0.076)+vec3(0.05,0.06,0.072)*eyeUv.y;
      vec3 base=mix(ground,sky,horizon);
      float wall=step(0.68,eyeUv.x)*step(0.14,eyeUv.y)*step(eyeUv.y,0.81);
      float mortar=step(0.97,fract(eyeUv.y*48.0))*0.065;
      base=mix(base,vec3(0.14,0.12,0.14)-mortar,wall);
      float lamp=exp(-24.0*distance(eyeUv,vec2(0.34,0.58)));
      base+=vec3(0.21,0.13,0.04)*lamp;
      color=vec4(base,1.0);
    }
  `;
  const windFragment = `#version 300 es
    precision highp float;precision highp int;
    uniform float uTime;
    layout(location=0) out vec2 field;
    void main(){
      int index=int(gl_FragCoord.x), z=int(gl_FragCoord.y);
      int x=index/32, phaseBin=index%32;
      float xCenter=-128.0+(float(x)+0.5)*4.0;
      float zCenter=-128.0+(float(z)+0.5)*4.0;
      float phase=(float(phaseBin)+0.5)*(6.2831853/32.0);
      field=vec2(sin(phase+uTime*0.85+zCenter*0.035),
                 cos(phase*0.73+uTime*0.58+xCenter*0.025));
    }
  `;

  function shader(type, source) {
    const result=gl.createShader(type);
    gl.shaderSource(result,source);gl.compileShader(result);
    if(!gl.getShaderParameter(result,gl.COMPILE_STATUS))
      throw new Error('Compilation GLSL : '+gl.getShaderInfoLog(result));
    return result;
  }
  function program(vertex,fragment,varyings) {
    const p=gl.createProgram();
    gl.attachShader(p,shader(gl.VERTEX_SHADER,vertex));
    gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fragment));
    if(varyings)gl.transformFeedbackVaryings(p,varyings,gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error('Liaison GLSL : '+gl.getProgramInfoLog(p));
    return p;
  }

  let updateProgram, snowProgram, backdropProgram, windProgram;
  let windTexture=null, windFbo=null, windCpuFallback=null;
  let buffers=[], vaos=[], source=0, initialData=null, count=0;
  let activeMode='baseline', simulationTime=0, lastFrame=0, frames=0;
  let wallSamples=[], gpuTime=null, queryExtension=null, queries=[];
  let ongoingTest=null, testing=false, stopped=false;

  function makeVao(buffer) {
    const vao=gl.createVertexArray();gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,4,gl.FLOAT,false,32,0);
    gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,4,gl.FLOAT,false,32,16);
    gl.bindVertexArray(null);return vao;
  }
  function mix(h){
    h^=h>>>16;h=Math.imul(h,0x7feb352d);h^=h>>>15;
    h=Math.imul(h,0x846ca68b);return (h^(h>>>16))>>>0;
  }
  function fraction(h){return (mix(h)&0xffffff)/16777216;}
  function seedStates(n){
    const result=new Float32Array(n*8), seed=0x76a019ab;
    for(let i=0;i<n;i++){
      const key=mix((i^seed)>>>0), off=i*8;
      const phase=fraction((key+67)>>>0), fall=.65+1.15*fraction((key+79)>>>0);
      result[off]=(fraction((key+11)>>>0)-.5)*256;
      result[off+1]=.04+fraction((key+29)>>>0)*17.96;
      result[off+2]=(fraction((key+47)>>>0)-.5)*256;
      result[off+3]=phase;
      result[off+4]=(fraction((key+73)>>>0)-.5)*.35;
      result[off+5]=-fall;
      result[off+6]=(fraction((key+97)>>>0)-.5)*.35;
      result[off+7]=.5+fraction((key+103)>>>0);
    }
    return result;
  }
  function releaseBuffers(){
    for(const vao of vaos)gl.deleteVertexArray(vao);
    for(const buffer of buffers)gl.deleteBuffer(buffer);
    buffers=[];vaos=[];
  }
  function resetState(n){
    if(n!==count||!initialData){
      releaseBuffers();count=n;initialData=seedStates(n);
      for(let i=0;i<2;i++){
        const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
        gl.bufferData(gl.ARRAY_BUFFER,i===0?initialData:initialData.byteLength,gl.DYNAMIC_COPY);
        buffers.push(buffer);vaos.push(makeVao(buffer));
      }
    }else{
      gl.bindBuffer(gl.ARRAY_BUFFER,buffers[0]);
      gl.bufferSubData(gl.ARRAY_BUFFER,0,initialData);
    }
    source=0;simulationTime=0;
    countElement.textContent=count.toLocaleString('fr-CA');
    if(gl.getError()===gl.OUT_OF_MEMORY)throw new Error('Mémoire GPU insuffisante pour ce nombre de particules');
  }

  function prepareWind(){
    windTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,windTexture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RG32F,TEX_WIDTH,GRID,0,gl.RG,gl.FLOAT,null);
    if(gl.getExtension('EXT_color_buffer_float')){
      windFbo=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,windFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,windTexture,0);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE){
        gl.deleteFramebuffer(windFbo);windFbo=null;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    }
    if(!windFbo)windCpuFallback=new Float32Array(TEX_WIDTH*GRID*2);
  }
  function updateWindGrid(time){
    if(windFbo){
      gl.bindFramebuffer(gl.FRAMEBUFFER,windFbo);
      gl.viewport(0,0,TEX_WIDTH,GRID);gl.useProgram(windProgram);
      gl.uniform1f(gl.getUniformLocation(windProgram,'uTime'),time);
      gl.bindVertexArray(null);gl.drawArrays(gl.TRIANGLES,0,3);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    }else{
      let i=0;
      for(let z=0;z<GRID;z++)for(let x=0;x<GRID;x++)for(let phase=0;phase<PHASES;phase++){
        const p=(phase+.5)/PHASES*6.2831853;
        windCpuFallback[i++]=Math.sin(p+time*.85+(-128+(z+.5)*4)*.035);
        windCpuFallback[i++]=Math.cos(p*.73+time*.58+(-128+(x+.5)*4)*.025);
      }
      gl.bindTexture(gl.TEXTURE_2D,windTexture);
      gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,TEX_WIDTH,GRID,gl.RG,gl.FLOAT,windCpuFallback);
    }
  }

  function step(dt){
    simulationTime+=dt;
    const cells=activeMode==='cells'||activeMode==='combined';
    if(cells)updateWindGrid(simulationTime);
    gl.useProgram(updateProgram);
    gl.uniform1f(gl.getUniformLocation(updateProgram,'uDt'),dt);
    gl.uniform1f(gl.getUniformLocation(updateProgram,'uTime'),simulationTime);
    gl.uniform1i(gl.getUniformLocation(updateProgram,'uCellWind'),cells?1:0);
    gl.uniform1i(gl.getUniformLocation(updateProgram,'uCells'),0);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,windTexture);
    gl.bindVertexArray(vaos[source]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,buffers[1-source]);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS,0,count);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    // Release the transform-feedback binding before using this buffer as a
    // vertex source in the next pass. WebGL forbids simultaneous feedback.
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,null);
    source=1-source;
  }

  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.viewport(0,0,resolution[0],resolution[1]);gl.disable(gl.BLEND);
    gl.bindVertexArray(null);gl.useProgram(backdropProgram);
    gl.uniform2f(gl.getUniformLocation(backdropProgram,'uResolution'),resolution[0],resolution[1]);
    gl.drawArrays(gl.TRIANGLES,0,3);
    gl.useProgram(snowProgram);gl.bindVertexArray(vaos[source]);
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    const batched=activeMode==='stereo'||activeMode==='combined';
    gl.uniform1i(gl.getUniformLocation(snowProgram,'uBatched'),batched?1:0);
    if(batched)gl.drawArraysInstanced(gl.POINTS,0,count,2);
    else for(let eye=0;eye<2;eye++){
      gl.uniform1i(gl.getUniformLocation(snowProgram,'uEye'),eye);
      gl.drawArrays(gl.POINTS,0,count);
    }
    gl.disable(gl.BLEND);gl.bindVertexArray(null);
  }

  function gpuQueryStart(){
    if(!queryExtension||queries.length>2||frames%12!==0)return null;
    const q=gl.createQuery();gl.beginQuery(queryExtension.TIME_ELAPSED_EXT,q);return q;
  }
  function gpuQueryEnd(q){if(q){gl.endQuery(queryExtension.TIME_ELAPSED_EXT);queries.push(q);}}
  function pollGpuTime(){
    if(!queryExtension||!queries.length)return;
    const q=queries[0];
    if(!gl.getQueryParameter(q,gl.QUERY_RESULT_AVAILABLE))return;
    const disjoint=gl.getParameter(queryExtension.GPU_DISJOINT_EXT);
    if(!disjoint)gpuTime=gl.getQueryParameter(q,gl.QUERY_RESULT)/1e6;
    gl.deleteQuery(q);queries.shift();
  }
  function setMode(name){
    activeMode=name;
    for(const button of modeButtons){
      const active=button.dataset.mode===name;
      button.classList.toggle('active',active);
      button.setAttribute('aria-pressed',active?'true':'false');
    }
  }
  function updateHud(now, frameSeconds){
    if(frameSeconds>0&&frameSeconds<.5){
      wallSamples.push(frameSeconds);
      if(wallSamples.length>50)wallSamples.shift();
    }
    if(frames%9!==0)return;
    const mean=wallSamples.length?wallSamples.reduce((a,b)=>a+b,0)/wallSamples.length:0;
    fpsElement.textContent=mean?(1/mean).toFixed(1):'—';
    frameElement.textContent=mean?(mean*1000).toFixed(1)+' ms':'— ms';
    gpuElement.textContent=gpuTime!==null?gpuTime.toFixed(2)+' ms':'N/D';
    status.textContent=labels[activeMode]+' · '+(windFbo?'vent GPU':'vent CPU (secours)');
  }

  function tick(now){
    if(stopped)return;
    try {
      const frameSeconds=lastFrame?(now-lastFrame)/1000:1/60;
      lastFrame=now;
      const dt=Math.min(.05,Math.max(.001,frameSeconds));
      const q=gpuQueryStart();
      step(dt);render();gpuQueryEnd(q);pollGpuTime();
      frames++;updateHud(now,frameSeconds);
      if(ongoingTest){
        const t=ongoingTest;
        if(now>=t.warmUntil&&now<t.endAt&&frameSeconds>.001&&frameSeconds<.5){
          t.durations.push(frameSeconds);
          if(gpuTime!==null)t.gpu.push(gpuTime);
        }
        if(now>=t.endAt){ongoingTest=null;t.resolve({frames:t.durations.length,
          fps:t.durations.length/t.durations.reduce((a,b)=>a+b,0),
          frameMs:1000*t.durations.reduce((a,b)=>a+b,0)/t.durations.length,
          gpuMs:t.gpu.length?t.gpu.reduce((a,b)=>a+b,0)/t.gpu.length:null});}
      }
    } catch(e){
      stopped=true;status.textContent='Erreur de rendu';benchmarkStatus.textContent=e.message;
      console.error(e);return;
    }
    requestAnimationFrame(tick);
  }

  function compareResults(results){
    const base=results.baseline;
    resultsBody.textContent='';
    for(const name of modes){
      const entry=results[name];
      const row=document.createElement('tr');
      if(name!=='baseline'&&entry.fps>base.fps)row.className='best';
      for(const value of [labels[name],entry.fps.toFixed(1),entry.frameMs.toFixed(1),
          entry.gpuMs===null?'N/D':entry.gpuMs.toFixed(2),
          name==='baseline'?'1,00×':(entry.fps/base.fps).toFixed(2).replace('.',',')+'×']){
        const td=document.createElement('td');td.textContent=value;row.appendChild(td);
      }
      resultsBody.appendChild(row);
    }
  }
  async function benchmarkAll(){
    if(testing||stopped)return;
    testing=true;benchmarkButton.disabled=true;
    select.disabled=true;resolutionSelect.disabled=true;modeButtons.forEach(b=>b.disabled=true);
    const originalMode=activeMode;
    const results={};
    try {
      for(let i=0;i<modes.length;i++){
        const name=modes[i];setMode(name);
        benchmarkBadge.textContent=`MESURE ${i+1} / ${modes.length}`;
        benchmarkStatus.textContent=`${labels[name]} : préchauffage 0,5 s puis mesure 2 s…`;
        resetState(count);wallSamples=[];lastFrame=0;gpuTime=null;
        results[name]=await new Promise(resolve=>{
          const now=performance.now();
          ongoingTest={resolve,warmUntil:now+500,endAt:now+2500,durations:[],gpu:[]};
        });
      }
      compareResults(results);
      benchmarkBadge.textContent='TERMINÉ';
      benchmarkStatus.textContent=`4 modes mesurés sur ${count.toLocaleString('fr-CA')} particules · ${resolution[1]} × ${resolution[1]} par œil · 2 secondes chacun · même état initial.`;
    }catch(e){
      benchmarkBadge.textContent='ERREUR';benchmarkStatus.textContent=e.message;
    }finally{
      setMode(originalMode);testing=false;benchmarkButton.disabled=false;
      select.disabled=false;resolutionSelect.disabled=false;modeButtons.forEach(b=>b.disabled=false);
    }
  }

  try {
    updateProgram=program(updateSource,`#version 300 es\nprecision mediump float;out vec4 discardColor;void main(){discardColor=vec4(0.0);}`,['nextPosition','nextVelocity']);
    snowProgram=program(renderSource,renderFragment);
    backdropProgram=program(fullScreenVertex,backdropFragment);
    windProgram=program(fullScreenVertex,windFragment);
    prepareWind();resetState(Number(select.value));
    queryExtension=gl.getExtension('EXT_disjoint_timer_query_webgl2');
    for(const button of modeButtons)button.addEventListener('click',()=>{if(!testing)setMode(button.dataset.mode);});
    select.addEventListener('change',()=>{
      if(testing)return;
      try {resetState(Number(select.value));wallSamples=[];}
      catch(e){status.textContent=e.message;console.error(e);}
    });
    resolutionSelect.addEventListener('change',()=>{
      if(testing)return;
      const perEye=Number(resolutionSelect.value);
      canvas.width=resolution[0]=perEye*2;
      canvas.height=resolution[1]=perEye;
      resolutionHud.textContent=`${perEye*2} × ${perEye} pixels au total · ${perEye} × ${perEye} / œil`;
      wallSamples=[];gpuTime=null;
    });
    benchmarkButton.addEventListener('click',benchmarkAll);
    requestAnimationFrame(tick);
  } catch(e){
    status.textContent='Erreur WebGL2';benchmarkStatus.textContent=e.message;
    benchmarkButton.disabled=true;console.error(e);
  }
})();
