/* The 3D scene gallery. Loads the browser models of the ten rooms already shown in the room explorer
   (scenes/assets, see README) and lets a visitor orbit them, open their moving parts and find each task. */
import { buildRigidReplay } from './rigid-replay.js?v=20261113035012';
import * as THREE from 'three';
import { OrbitControls } from '../scenes/vendor/OrbitControls.js';
import { GLTFLoader } from '../scenes/vendor/GLTFLoader.js';
import { DRACOLoader } from '../scenes/vendor/DRACOLoader.js';
import { RoomEnvironment } from '../scenes/vendor/RoomEnvironment.js';

const BASE = 'scenes/';
const $ = (s, el) => (el || document).querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const nice = id => String(id).replace(/__\d+$/, '').replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
const smooth = t => t * t * (3 - 2 * t);

const WALLS_ON = new Set();   // scenes that open with their walls up (never the roof)
const HIDDEN = new Set(['living-room-pool-013', 'humanoid-test-lab-001']);
const REMOVED = { 'biology-lab': ['pick_place_1', 'push_1'] };   // tasks left out of a scene's list
const TABPIC = { 'biology-lab': 'assets/rooms/biology-lab/view-1.webp' };   // the first picture of this scene is mostly ceiling
const TITLES = { 'f6-commercial-kitchen-001': 'Commercial kitchen' };

export function createViewer(canvas, stage, onStatus, wheelZoom = true) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(39, 1, 0.1, 300);
  const pmrem = new THREE.PMREMGenerator(renderer), env = new RoomEnvironment();
  scene.environment = pmrem.fromScene(env, 0.04).texture;
  scene.environmentIntensity = 0.6;
  env.dispose(); pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xfff9e7, 0x797469, 0.65));
  const sun = new THREE.DirectionalLight(0xfff0d6, 2.5);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0003; sun.shadow.normalBias = 0.008; sun.shadow.radius = 3;
  const fill = new THREE.DirectionalLight(0xe2edff, 0.45); fill.position.set(4, 4, -5);
  scene.add(sun, sun.target, fill);

  const controls = new OrbitControls(camera, canvas);
  // Where wheelZoom is off (the two-pane players), the wheel scrolls the page; Ctrl or Cmd + wheel still zooms.
  if (!wheelZoom) canvas.addEventListener('wheel', e => { if (!e.ctrlKey && !e.metaKey) e.stopImmediatePropagation(); }, { capture: true, passive: true });
  controls.enableDamping = true; controls.dampingFactor = 0.08; controls.maxPolarAngle = Math.PI * 0.495;
  controls.autoRotateSpeed = 0.7;
  controls.addEventListener('start', () => { controls.autoRotate = false; stage.classList.add('touched'); });

  const draco = new DRACOLoader().setDecoderPath(BASE + 'vendor/draco/');
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  let room = null, pad = null, info = null, meshes = [], selection = [], boxes = [], mixer = null;
  let states = [], tweens = [], transition = null, serial = 0, visible = true, walls = false;
  let replay = null, replaySerial = 0;
  let bounds = new THREE.Box3(new THREE.Vector3(-3, 0, -3), new THREE.Vector3(3, 2.6, 3));

  const cut = new THREE.Plane(new THREE.Vector3(0, -1, 0), 2.4);
  renderer.localClippingEnabled = true;
  const metrics = () => { const c = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3()); return { c, size, span: Math.max(size.x, size.z, 2) }; };

  function moveCamera(position, target, animate = true) {
    if (animate) transition = { from: camera.position.clone(), to: new THREE.Vector3(...position), tFrom: controls.target.clone(), tTo: new THREE.Vector3(...target), start: performance.now() };
    else { transition = null; camera.position.set(...position); controls.target.set(...target); controls.update(); }
  }
  function setView(view, animate = true) {
    const { c, size, span } = metrics(), fit = Math.max(1, 1 / camera.aspect), floor = bounds.min.y;
    const wide = camera.aspect > 1.4 ? 0.74 : 1;   // a wide stage is limited by its height, so stand closer
    const target = [c.x, floor + Math.min(size.y * 0.23, 0.85), c.z];
    const presets = {
      angled: [[c.x + span * 0.8 * fit * wide, floor + span * 0.86 * fit * wide, c.z + span * 1.02 * fit * wide], target],
      top: [[c.x, floor + span * 1.95 * fit, c.z + 0.001], [c.x, floor, c.z]],
      inside: [[c.x + size.x * 0.30, floor + Math.min(1.7, size.y * 0.65), c.z + size.z * 0.33], [c.x - size.x * 0.18, floor + Math.min(1.1, size.y * 0.4), c.z - size.z * 0.13]]
    };
    if (view === 'inside') setWalls(true);
    moveCamera(...presets[view], animate);
  }
  function setWalls(on, shell = on) {
    walls = on;
    cut.constant = bounds.min.y + Math.min(2.4, (bounds.max.y - bounds.min.y) * 0.8);
    if (!room) return;
    room.traverse(o => {
      if (!o.isMesh) return;
      const kind = o.userData.kind;
      if (info.mobility?.objects?.[o.userData.objectId]?.visualProxy) { o.visible = false; return; }
      o.visible = kind === 'walls' ? walls : kind === 'backdrop' ? shell : kind === 'ceiling' ? false : true;
      // Walls shown without the outer shell are cut off below the ceiling, so the room stays open from above.
      if (kind === 'walls') (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { m.clippingPlanes = shell ? null : [cut]; m.clipShadows = true; m.side = THREE.DoubleSide; m.needsUpdate = true; });
    });
    renderer.shadowMap.needsUpdate = true;
  }
  function clearSelection() {
    for (const m of selection) (Array.isArray(m.material) ? m.material : [m.material]).forEach(mat => mat.emissive?.set(0x000000));
    for (const b of boxes) { scene.remove(b); b.geometry.dispose(); b.material.dispose(); }
    selection = []; boxes = [];
  }
  function focusBounds(b) {
    if (b.isEmpty()) return;
    const { span } = metrics(), c = b.getCenter(new THREE.Vector3()), size = b.getSize(new THREE.Vector3()).length();
    const dir = camera.position.clone().sub(controls.target).normalize(); dir.y = Math.max(dir.y, 0.45); dir.normalize();
    const distance = Math.max(2, Math.min(span * 1.8, size * 1.65 + 1.2));
    moveCamera(c.clone().addScaledVector(dir, distance).toArray(), c.toArray());
  }
  function select(ids, focus) {
    clearSelection();
    const wanted = new Set(ids);
    selection = meshes.filter(m => wanted.has(m.userData.objectId) && !['floor', 'walls', 'ceiling', 'backdrop'].includes(m.userData.kind));
    if (!selection.length) return false;
    const all = new THREE.Box3();
    for (const m of selection) { (Array.isArray(m.material) ? m.material : [m.material]).forEach(mat => mat.emissive?.set(0x1d4a36)); all.expandByObject(m); }
    for (const id of ids) {
      const b = new THREE.Box3(); selection.filter(m => m.userData.objectId === id).forEach(m => b.expandByObject(m));
      if (b.isEmpty()) continue;
      b.expandByScalar(0.025);
      const helper = new THREE.Box3Helper(b, 0x48e0a4); helper.material.depthTest = false; helper.material.transparent = true; helper.material.opacity = 0.85;
      helper.userData.objectId = id; scene.add(helper); boxes.push(helper);
    }
    if (focus) { controls.autoRotate = false; setWalls(false); focusBounds(all); }
    return true;
  }
  function applyMotion(state, amount) {
    state.amount = THREE.MathUtils.clamp(amount, 0, 1);
    state.action.time = state.amount * state.duration;
  }
  function play(list, hold = 900) {
    const now = performance.now();
    for (const s of list) tweens.push({ s, from: s.amount, to: 1, start: now, dur: 1400, back: hold });
  }
  function tick(now) {
    if (transition) {
      const t = Math.min((now - transition.start) / 900, 1), e = smooth(t);
      camera.position.lerpVectors(transition.from, transition.to, e); controls.target.lerpVectors(transition.tFrom, transition.tTo, e);
      if (t === 1) transition = null;
    }
    if (tweens.length) {
      const keep = [];
      for (const t of tweens) {
        if (now < t.start) { keep.push(t); continue; }
        const p = Math.min((now - t.start) / t.dur, 1);
        applyMotion(t.s, t.from + (t.to - t.from) * smooth(p));
        if (p < 1) keep.push(t);
        else if (t.back != null) keep.push({ s: t.s, from: 1, to: t.s.part.initialFraction || 0, start: now + t.back, dur: 1400, back: null });
      }
      tweens = keep;
      mixer.update(0); room.updateMatrixWorld(true);
      for (const h of boxes) { h.box.makeEmpty(); meshes.filter(m => m.userData.objectId === h.userData.objectId).forEach(m => h.box.expandByObject(m)); h.box.expandByScalar(0.025); }
      renderer.shadowMap.needsUpdate = true;
    }
  }
  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(stage);
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || !room) return;
    tick(now); stepReplay(now); controls.update(); renderer.render(scene, camera);
  }
  resize(); requestAnimationFrame(frame);

  /* Recorded task replays: either body tracks played on the room's own meshes, or an animated overlay model. */
  function stopReplay() {
    replaySerial++;
    if (!replay) return;
    const r = replay; replay = null;
    r.mixer.stopAllAction(); r.mixer.uncacheRoot(r.model); scene.remove(r.model);
    if (!r.shared) dispose(r.model);
    for (const [o, v] of r.originals) o.visible = v;
    renderer.shadowMap.needsUpdate = true;
    onStatus({ replay: null });
  }
  function showReplay(t) {
    const r = replay; r.time = Math.max(0, Math.min(t, r.duration));
    r.action.time = r.time; r.mixer.update(0); renderer.shadowMap.needsUpdate = true;
    onStatus({ replay: { time: r.time, duration: r.duration, playing: r.playing, title: r.title, caption: r.caption } });
  }
  function stepReplay(now) {
    if (!replay?.playing) return;
    const t = (now - replay.startedAt) / 1000;
    if (t >= replay.duration) { replay.playing = false; showReplay(replay.duration); } else showReplay(t);
  }
  async function playReplay(meta) {
    stopReplay(); clearSelection(); tweens = [];
    const mine = replaySerial;
    onStatus({ replay: { loading: true, title: meta.title } });
    try {
      const res = await fetch(BASE + meta.file);
      if (!res.ok) throw Error('This recording is not in the site folder.');
      const bytes = await res.arrayBuffer();
      let built, shared = false;
      if (new DataView(bytes).getUint32(0, false) === 0x676c5446) built = await new GLTFLoader().setDRACOLoader(draco).parseAsync(bytes, '');
      else { const motion = JSON.parse(new TextDecoder().decode(bytes)); built = buildRigidReplay(motion, room, motion.taskId); shared = true; }
      if (mine !== replaySerial || !room) { if (!shared) dispose(built.scene); return; }
      controls.autoRotate = false; setWalls(false);
      const originals = new Map(), ids = new Set(built.scene.userData.hideObjectIds || []);
      for (const o of built.hidden || []) originals.set(o, o.visible);
      room.traverse(o => { if (o.isMesh && ids.has(o.userData.objectId)) originals.set(o, o.visible); });
      for (const o of originals.keys()) o.visible = false;
      built.scene.traverse(o => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });
      const clip = built.animations[0], mixer2 = new THREE.AnimationMixer(built.scene), action = mixer2.clipAction(clip);
      action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; action.play(); action.paused = true;
      scene.add(built.scene);
      const target = new THREE.Vector3(...meta.target), off = new THREE.Vector3(...meta.eye).sub(target).multiplyScalar(Math.max(1, 1 / camera.aspect));
      moveCamera(target.clone().add(off).toArray(), target.toArray());
      replay = { model: built.scene, mixer: mixer2, action, originals, shared, duration: clip.duration, time: 0, playing: true, startedAt: performance.now() + 700, title: meta.title, caption: meta.caption };
      showReplay(0);
    } catch (e) {
      if (mine !== replaySerial) return;
      console.error(e); onStatus({ replay: { error: e.message || 'The recording could not load.', title: meta.title } });
    }
  }

  function dispose(obj) {
    obj.traverse(o => {
      if (!o.isMesh) return;
      o.geometry.dispose();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => { for (const k in m) if (m[k]?.isTexture) m[k].dispose(); m.dispose(); });
    });
  }
  function clearRoom() {
    stopReplay(); clearSelection(); tweens = []; states = [];
    if (room) { scene.remove(room); dispose(room); room = null; }
    if (pad) { scene.remove(pad); pad.geometry.dispose(); pad.material.dispose(); pad = null; }
    meshes = []; mixer = null;
  }
  async function fetchModel(meta, mine) {
    const total = meta.modelBytes, bytes = new Uint8Array(total); let offset = 0;
    for (const path of meta.modelParts) {
      const res = await fetch(BASE + path);
      if (!res.ok) throw Error('A part of this scene is unavailable.');
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (mine !== serial) { reader.cancel(); throw new DOMException('changed', 'AbortError'); }
        bytes.set(value, offset); offset += value.byteLength;
        onStatus({ loading: offset / total });
      }
    }
    if (meta.modelEncoding !== 'gzip') return bytes.buffer;
    return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  async function load(entry) {
    const mine = ++serial;
    clearRoom(); onStatus({ loading: 0 });
    try {
      const meta = await (await fetch(BASE + entry.metadata)).json();
      if (mine !== serial) return null;
      const buffer = await fetchModel(meta, mine);
      onStatus({ loading: 1, preparing: true });
      const gltf = await new GLTFLoader().setDRACOLoader(draco).parseAsync(buffer, '');
      if (mine !== serial) { dispose(gltf.scene); return null; }
      info = meta; room = gltf.scene;
      room.userData.decodedModelSha256 = info.decodedModelSha256;
      room.updateMatrixWorld(true);
      room.traverse(o => {
        const index = gltf.parser.associations.get(o)?.nodes;
        if (Number.isInteger(index)) o.userData.replayNodeIndex = index;
        if (o.isMesh) o.userData.replayBindWorld = o.matrixWorld.toArray();
      });
      const originals = new Set();
      room.traverse(o => {
        if (!o.isMesh) return;
        let parent = o; while (parent && !parent.userData.objectId) parent = parent.parent;
        const id = parent?.userData.objectId || o.name.replace(/_\d+$/, '');
        o.userData.objectId = id; o.userData.kind = info.objects[id] || 'object';
        const setup = original => { originals.add(original); const m = original.clone(); m.emissive?.set(0); if (m.map) m.map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
          return m; };
        o.material = Array.isArray(o.material) ? o.material.map(setup) : setup(o.material);
        o.castShadow = true; o.receiveShadow = true; meshes.push(o);
      });
      originals.forEach(m => m.dispose());
      // Some models lay one part exactly on another (a coloured top on a pale body, a panel on a door). Two surfaces
      // at the same depth flicker as the view turns. So within each object the parts are ranked by size and each is
      // pushed back a little more than the next smaller one: the smaller part, the one laid on top, always wins.
      const groups = new Map(), box3 = new THREE.Box3(), dim = new THREE.Vector3();
      for (const o of meshes) {
        box3.setFromObject(o).getSize(dim);
        o.userData.bulk = Math.max(dim.x, 1e-3) * Math.max(dim.y, 1e-3) * Math.max(dim.z, 1e-3);
        if (!groups.has(o.userData.objectId)) groups.set(o.userData.objectId, []);
        groups.get(o.userData.objectId).push(o);
      }
      for (const list of groups.values()) {
        list.sort((p, q) => p.userData.bulk - q.userData.bulk);
        list.forEach((o, i) => {
          const k = Math.min(i, 24) * 0.75;
          for (const m of Array.isArray(o.material) ? o.material : [o.material]) { m.polygonOffset = k > 0; m.polygonOffsetFactor = k; m.polygonOffsetUnits = k * 4; }
        });
      }
      scene.add(room);
      const b = info.bounds;
      bounds = b?.min?.length === 3 ? new THREE.Box3(new THREE.Vector3(...b.min), new THREE.Vector3(...b.max)) : new THREE.Box3().setFromObject(room);
      const { c, size, span } = metrics(), floor = bounds.min.y;
      controls.maxDistance = Math.max(22, span * 4); controls.minDistance = 0.6;
      camera.far = Math.max(150, span * 12); camera.updateProjectionMatrix();
      pad = new THREE.Mesh(new THREE.BoxGeometry(size.x + 0.42, 0.10, size.z + 0.42), new THREE.MeshStandardMaterial({ color: 0x2a3038, roughness: 1 }));
      pad.position.set(c.x, floor - 0.14, c.z); pad.receiveShadow = true; scene.add(pad);
      sun.position.set(c.x - span * 0.55, floor + span * 1.35, c.z - span * 0.75); sun.target.position.set(c.x, floor, c.z);
      Object.assign(sun.shadow.camera, { left: -span * 0.8, right: span * 0.8, top: span * 0.8, bottom: -span * 0.8, near: 0.1, far: span * 4 + 10 });
      sun.shadow.camera.updateProjectionMatrix();
      mixer = new THREE.AnimationMixer(room);
      for (const part of info.articulations || []) {
        const clip = THREE.AnimationClip.findByName(gltf.animations, part.clip);
        if (!clip) continue;
        const action = mixer.clipAction(clip); action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true; action.play(); action.paused = true;
        const s = { part, action, duration: clip.duration, amount: 0 };
        applyMotion(s, part.initialFraction || 0); states.push(s);
      }
      mixer.update(0);
      setWalls(WALLS_ON.has(info.id), false); setView('angled', false);
      controls.autoRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
      stage.classList.remove('touched');
      renderer.shadowMap.needsUpdate = true;
      onStatus({ ready: true, info, parts: states.length });
      return info;
    } catch (e) {
      if (e.name === 'AbortError' || mine !== serial) return null;
      console.error(e); onStatus({ error: e.message || 'The scene could not load.' });
      return null;
    }
  }

  canvas.addEventListener('click', e => {
    if (!room || stage.classList.contains('dragging')) return;
    const r = canvas.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(meshes.filter(m => m.visible), false).find(h => !['floor', 'walls', 'ceiling', 'backdrop'].includes(h.object.userData.kind));
    if (!hit) { clearSelection(); onStatus({ picked: null }); return; }
    const id = hit.object.userData.objectId;
    select([id], false);
    const mine = states.filter(s => (s.part.objectId || s.part.id) === id);
    if (mine.length) play(mine.slice(0, 12));
    const status = info.mobility?.objects?.[id]?.status;
    onStatus({ picked: { name: nice(id), detail: mine.length ? mine.length + (mine.length > 1 ? ' moving parts' : ' moving part') : status === 'fixed' ? 'fixed to the building' : status === 'free' ? 'free to move' : '' } });
  });
  let down = null;
  canvas.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; stage.classList.remove('dragging'); });
  canvas.addEventListener('pointermove', e => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) stage.classList.add('dragging'); });
  addEventListener('pointerup', () => { down = null; setTimeout(() => stage.classList.remove('dragging'), 0); });

  return {
    load, setView, setWalls, clearSelection, playReplay, stopReplay,
    zoom(f) { const t = controls.target, p = camera.position.clone().sub(t).multiplyScalar(f).add(t); moveCamera(p.toArray(), t.toArray()); },
    toggleReplay() { if (!replay) return; if (replay.time >= replay.duration) replay.time = 0; replay.playing = !replay.playing; replay.startedAt = performance.now() - replay.time * 1000; showReplay(replay.time); },
    seekReplay(f) { if (!replay) return; replay.playing = false; showReplay(f * replay.duration); },
    get walls() { return walls; },
    spin() { controls.autoRotate = !controls.autoRotate; return controls.autoRotate; },
    openAll(hold = 1100) { if (states.length) { clearSelection(); tweens = []; play(states, hold); } },
    surprise() {
      if (!states.length) return null;
      const s = states[Math.floor(Math.random() * states.length)], id = s.part.objectId || s.part.id;
      const group = states.filter(x => (x.part.objectId || x.part.id) === id).slice(0, 12);
      select([id], true); setTimeout(() => play(group, 1200), 700);
      return nice(id);
    },
    locate(task) { return select(task.objects || [], true); }
  };
}

/* ---------- Gallery UI ---------- */
async function gallery() {
  const root = $('#gallery');
  if (!root) return;
  let catalog;
  try { catalog = (await (await fetch(BASE + 'assets/rooms.json')).json()).rooms; }
  catch (e) { root.classList.add('missing'); return; }

  // Shown here: eight of the ten explorer scenes, with plain names.
  catalog = catalog.filter(r => !HIDDEN.has(r.id)).map(r => ({ ...r, title: TITLES[r.id] || r.title }));
  const stage = $('.stage', root), canvas = $('canvas', root), chips = $('.roomchips', root), panel = $('.sceneinfo', root), film = $('.coverfilm', root);
  const cover = $('.cover', root), status = $('.status', root), picked = $('.picked', root);
  let viewer = null, current = catalog[0], info = null, started = false, replays = {}, playlist = [], queue = -1, nextTimer = 0, openTimer = 0, wantTasks = false, filmOn = false;
  const bar = $('.replaybar', root), fmtT = t => Math.floor(t / 60) + ':' + String(Math.floor(t) % 60).padStart(2, '0');

  chips.innerHTML = catalog.map(r =>
    `<button type="button" data-id="${r.id}" aria-pressed="${r.id === current.id}" style="background-image:url(${BASE + (TABPIC[r.id] || r.thumbnail)})"><span>${esc(r.title)}</span></button>`).join('');

  function drawPanel() {
    const r = current, tasks = info?.tasks || [];
    // The task film shows the tasks done by hand forces. The robot's tasks are left out.
    const gone = new Set(REMOVED[r.id] || []);
    playlist = [];
    const order = tasks.map((t, i) => i).filter(i => !gone.has(tasks[i].id) && tasks[i].kind !== 'Robot');
    panel.innerHTML =
      `<p class="lede">${esc(r.description.replace(/\s*$/, ''))} It holds ${r.objects} objects and ${r.articulations} moving parts.</p>`;
    playlist = order.filter(i => replays[tasks[i].id]);
  }
  function setCover() {
    cover.style.backgroundImage = 'none'; film.poster = BASE + current.thumbnail; film.hidden = false;
    if (filmOn) { film.src = BASE + current.film; film.play().catch(() => {}); }
    $('.cover b', root).textContent = current.title;
    $('.cover small', root).textContent = '';
    cover.hidden = false;
  }
  function onStatus(s) {
    if (s.error) { status.hidden = false; status.textContent = s.error; return; }
    if (s.loading != null && !s.ready) { status.hidden = false; status.textContent = s.preparing ? 'Unpacking geometry and textures…' : 'Loading the scene · ' + Math.round(s.loading * 100) + '%'; }
    if (s.ready) {
      status.hidden = true; cover.hidden = true; film.pause(); film.hidden = true; stage.classList.add('live'); info = s.info; drawPanel();
      // on arrival every door and drawer opens, holds for three seconds, and closes again
      clearTimeout(openTimer); openTimer = setTimeout(() => { if (viewer && queue < 0) viewer.openAll(3000); }, 900);
    }
    if ('replay' in s) {
      const r = s.replay; bar.hidden = !r; stage.classList.toggle('replaying', !!r);
      if (r) {
        $('.rtitle', bar).textContent = (queue >= 0 ? `Task ${queue + 1} of ${playlist.length}` : r.title) + (r.loading ? ' · loading…' : r.error ? ' · ' + r.error : '');
        $('.rtime', bar).textContent = r.duration ? fmtT(r.time) + ' / ' + fmtT(r.duration) : '';
        $('.rplay', bar).textContent = r.playing ? 'Pause' : r.duration && r.time >= r.duration ? 'Replay' : 'Play';
        $('.rplay', bar).disabled = !r.duration;
        if (r.duration) $('.rseek', bar).value = String(Math.round(r.time / r.duration * 1000));
        if (queue >= 0) $('.rcap', bar).textContent = info.tasks[playlist[queue]].text; else if (r.caption) $('.rcap', bar).textContent = r.caption;
        // the task film: when one task ends, the next starts by itself
        if (queue >= 0 && r.duration && !r.playing && r.time >= r.duration && !nextTimer) nextTimer = setTimeout(() => { nextTimer = 0; if (queue >= 0) playTask(queue + 1); }, 1100);
      } else { queue = -1; clearTimeout(nextTimer); nextTimer = 0; }
      markMode();
    }
    if ('picked' in s) { picked.hidden = !s.picked; if (s.picked) picked.innerHTML = `<b>${esc(s.picked.name)}</b>${s.picked.detail ? ' · ' + esc(s.picked.detail) : ''}`; }
  }
  async function enter() {
    started = true; root.classList.add('started'); cover.classList.add('busy');
    if (!viewer) viewer = createViewer(canvas, stage, onStatus);
    stage.classList.remove('live'); info = null; picked.hidden = true; drawPanel();
    replays = {};
    const [loaded, rec] = await Promise.all([viewer.load(current), fetch(BASE + 'assets/rooms/' + current.id + '/replays.json').then(r => r.ok ? r.json() : {}).catch(() => ({}))]);
    if (loaded && loaded.id === current.id) { replays = rec; drawPanel(); $('[data-act="walls"]', root)?.setAttribute('aria-pressed', String(viewer.walls)); if (wantTasks && playlist.length) { clearTimeout(openTimer); playTask(0); } }
    wantTasks = false; markMode();
    cover.classList.remove('busy');
  }
  chips.addEventListener('click', e => {
    const b = e.target.closest('button[data-id]'); if (!b) return;
    current = catalog.find(r => r.id === b.dataset.id);
    chips.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
    info = null; drawPanel();
    if (started) enter(); else setCover();
  });
  cover.addEventListener('click', enter);
  $('.tools', root).addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b || !viewer) return;
    const v = b.dataset.view;
    if (v) { viewer.setView(v); if (v !== 'inside') viewer.setWalls(false); }
    if (b.dataset.act === 'walls') b.setAttribute('aria-pressed', String(!viewer.walls)), viewer.setWalls(!viewer.walls);
    if (b.dataset.act === 'openall') viewer.openAll(3000);
    if (b.dataset.zoom) viewer.zoom(+b.dataset.zoom);
    if (b.dataset.act === 'spin') b.setAttribute('aria-pressed', String(viewer.spin()));
  });
  const dialog = $('#film');
  const onAct = e => {
    const act = e.target.closest('[data-act]')?.dataset.act, task = e.target.closest('[data-task]');
    if (act === 'film') { const v = $('video', dialog); $('h3', dialog).textContent = current.title; v.src = BASE + current.film; dialog.showModal(); v.play().catch(() => {}); }
    if (act === 'open' && viewer) viewer.openAll();
    if (act === 'tasks' && !info) { wantTasks = true; enter(); return; }
    if (act === 'tasks' && viewer && info && playlist.length) { picked.hidden = true; playTask(0); if (matchMedia('(max-width: 900px)').matches) stage.scrollIntoView({ block: 'center' }); }
    if (task && viewer && info) {
      panel.querySelectorAll('[data-task]').forEach(x => x.classList.toggle('on', x === task));
      const t = info.tasks[+task.dataset.task];
      if (replays[t.id]) { picked.hidden = true; viewer.playReplay(replays[t.id]); }
      else if (viewer.locate(t)) onStatus({ picked: { name: t.locatorSteps?.[0]?.label || 'Task objects', detail: '' } });
      if (matchMedia('(max-width: 900px)').matches) stage.scrollIntoView({ block: 'center' });
    }
  };
  // The two buttons on top of the view: look around freely, or watch the scene's tasks as one film. Either way back at any time.
  const modes = $('.modes', root);
  function markMode() {
    const tasks = queue >= 0 || wantTasks, explore = !tasks && started;
    modes.querySelector('[data-mode="tasks"]').setAttribute('aria-pressed', String(tasks));
    modes.querySelector('[data-mode="explore"]').setAttribute('aria-pressed', String(explore));
  }
  modes.addEventListener('click', e => {
    const b = e.target.closest('[data-mode]'); if (!b) return;
    if (b.dataset.mode === 'tasks') {
      if (!info) { wantTasks = true; markMode(); enter(); return; }
      if (playlist.length) { picked.hidden = true; clearTimeout(openTimer); playTask(0); }
    } else {
      wantTasks = false;
      if (!started) enter(); else if (viewer && queue >= 0) { viewer.stopReplay(); viewer.setView('angled'); }
    }
    markMode();
  });
  // One film of every task in the scene, one after another, each with its own line of text under the picture.
  function playTask(i) {
    clearTimeout(nextTimer); nextTimer = 0;
    if (i >= playlist.length) { viewer.stopReplay(); return; }
    queue = i;
    $('.rdots', bar).innerHTML = playlist.map((t, k) => `<button type="button" data-q="${k}" class="${k === i ? 'on' : k < i ? 'done' : ''}" aria-label="Task ${k + 1}"></button>`).join('');
    const want = i;
    viewer.playReplay(replays[info.tasks[playlist[i]].id]).then(() => { if (queue === -1 && want === i) queue = -1; });
    queue = i;
  }
  bar.addEventListener('click', e => {
    if (!viewer) return;
    const q = e.target.closest('[data-q]'); if (q) { playTask(+q.dataset.q); return; }
    if (e.target.closest('.rplay')) viewer.toggleReplay();
    if (e.target.closest('.rclose')) viewer.stopReplay();
  });
  $('.rseek', bar).addEventListener('input', e => viewer && viewer.seekReplay(e.target.valueAsNumber / 1000));
  $('.zoomers', root).addEventListener('click', e => { const b = e.target.closest('[data-zoom]'); if (b && viewer) viewer.zoom(+b.dataset.zoom); });
  drawPanel(); setCover();
  // the cover film is fetched only once the explorer is near the screen
  new IntersectionObserver((es, io) => { if (es[0].isIntersecting) { io.disconnect(); filmOn = true; if (!started) setCover(); } }, { rootMargin: '200px' }).observe(root);
}

gallery();  // module scripts run after the document is parsed
