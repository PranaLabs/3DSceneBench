/* "Caught in the test": real copies from the pilot, drawn live from the entrant's own scene file.
   Each copy opens on the same camera as the input picture beside it. Data: data/cases.js and data/copies/<room>.json. */
import * as THREE from 'three';
import { OrbitControls } from '../scenes/vendor/OrbitControls.js';
import { createViewer } from './scene.js';

const $ = (s, el) => (el || document).querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const smooth = t => t * t * (3 - 2 * t);
const UP = new THREE.Matrix4().makeRotationX(-Math.PI / 2);   // the copies are z-up, three.js is y-up

function bytes(b64, Type) {
  const bin = atob(b64), a = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
  return new Type(a.buffer);
}

function buildCopy(data) {
  const root = new THREE.Group();
  root.matrixAutoUpdate = false; root.matrix.copy(UP);
  const geos = new Map(), mats = new Map(), meshGeo = [];
  const geometry = g => {
    const [a, b, c] = g.s, key = g.t + (g.t === 'mesh' ? g.m : g.s.join(','));
    if (geos.has(key)) return geos.get(key);
    let geo;
    if (g.t === 'box') geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    else if (g.t === 'sphere') geo = new THREE.SphereGeometry(a, 20, 14);
    else if (g.t === 'ellipsoid') geo = new THREE.SphereGeometry(1, 20, 14).scale(a, b, c);
    else if (g.t === 'cylinder') geo = new THREE.CylinderGeometry(a, a, 2 * b, 20).rotateX(Math.PI / 2);
    else if (g.t === 'capsule') geo = new THREE.CapsuleGeometry(a, 2 * b, 6, 14).rotateX(Math.PI / 2);
    else if (g.t === 'plane') geo = new THREE.PlaneGeometry(2 * (a || 30), 2 * (b || 30));
    else {
      if (!meshGeo[g.m]) {
        const m = data.meshes[g.m], geoM = new THREE.BufferGeometry();
        geoM.setAttribute('position', new THREE.BufferAttribute(bytes(m.v, Float32Array), 3));
        geoM.setIndex(new THREE.BufferAttribute(bytes(m.f, Uint32Array), 1));
        geoM.computeVertexNormals();
        meshGeo[g.m] = geoM;
      }
      geo = meshGeo[g.m];
    }
    geos.set(key, geo);
    return geo;
  };
  const byObject = new Map(), byBody = new Map();
  for (const g of data.geoms) {
    const key = g.c.join(',');
    const mesh = new THREE.Mesh(geometry(g), null);
    // Each object gets its own materials so one can be lit up without touching the others.
    const mkey = g.o + '|' + key;
    if (!mats.has(mkey)) {
      const m = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(g.c[0], g.c[1], g.c[2], THREE.SRGBColorSpace), roughness: 0.75, metalness: 0.05, side: THREE.DoubleSide });
      if (g.c[3] < 0.98) { m.transparent = true; m.opacity = Math.max(0.15, g.c[3]); m.depthWrite = false; }
      mats.set(mkey, m);
    }
    mesh.material = mats.get(mkey);
    const r = g.r;
    mesh.matrixAutoUpdate = false;
    mesh.matrix.set(r[0], r[1], r[2], g.p[0], r[3], r[4], r[5], g.p[1], r[6], r[7], r[8], g.p[2], 0, 0, 0, 1);
    root.add(mesh);
    mesh.userData.base = mesh.matrix.clone();
    // Walls and ceilings of the room itself: hidden when the view is from inside the room, as on the true-scene side.
    if (g.t === 'box' && (!g.o || /^(wall|walls|ceiling|roof|room_geometry)(_|$|\d)/i.test(g.o))) {
      const ext = [0, 1, 2].map(i => Math.abs(r[3 * i]) * g.s[0] + Math.abs(r[3 * i + 1]) * g.s[1] + Math.abs(r[3 * i + 2]) * g.s[2]);   // half extents along x, y, z (z is up)
      const thinUp = ext[2] < 0.2 && ext[0] > 0.8 && ext[1] > 0.8, tall = ext[2] > 0.7 && Math.min(ext[0], ext[1]) < 0.2 && Math.max(ext[0], ext[1]) > 0.8;
      if ((thinUp && g.p[2] > 1.6) || tall) mesh.userData.shell = true;
    }
    if (!byObject.has(g.o)) byObject.set(g.o, []);
    byObject.get(g.o).push(mesh);
    if (g.b) { if (!byBody.has(g.b)) byBody.set(g.b, []); byBody.get(g.b).push(mesh); }
  }
  return { root, byObject, byBody };
}

function createStage(canvas, stage) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdfe3e6);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 120);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9690, 1.5), new THREE.AmbientLight(0xffffff, 0.55));
  const lamp = new THREE.PointLight(0xfff4e2, 2.2, 0, 0.6); camera.add(lamp); scene.add(camera);
  const controls = new OrbitControls(camera, canvas);
  // The mouse wheel scrolls the page, as everywhere else; hold Ctrl or Cmd to zoom with it, or use the + and - buttons.
  canvas.addEventListener('wheel', e => { if (!e.ctrlKey && !e.metaKey) e.stopImmediatePropagation(); }, { capture: true, passive: true });
  controls.enableDamping = true; controls.dampingFactor = 0.1; controls.minDistance = 0.3; controls.maxDistance = 30;
  let copy = null, home = null, move = null, lit = [], box = null, visible = true, act = null, extras = [], rec = null, centre = null;
  const ease = t => t < 0 ? 0 : t > 1 ? 1 : smooth(t);

  function fit() { const s = stage.clientWidth; if (s) { renderer.setSize(s, stage.clientHeight, false); camera.aspect = s / stage.clientHeight; camera.updateProjectionMatrix(); } }
  new ResizeObserver(fit).observe(stage);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(stage);
  (function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || !copy) return;
    if (move) {
      const t = Math.min((now - move.start) / 900, 1), e = smooth(t);
      camera.position.lerpVectors(move.p0, move.p1, e); controls.target.lerpVectors(move.t0, move.t1, e);
      if (t === 1) move = null;
    }
    if (act) stepAct(now);
    if (rec) stepRec(now);
    controls.update(); renderer.render(scene, camera);
  })(performance.now());
  fit();

  function go(pos, target, animate) {
    if (!animate) { camera.position.copy(pos); controls.target.copy(target); controls.update(); move = null; return; }
    move = { p0: camera.position.clone(), p1: pos, t0: controls.target.clone(), t1: target, start: performance.now() };
  }
  /* Move one object by the amount the grader measured. Coordinates here are the copy's own (z up). */
  function centreOf(meshes) {
    const c = new THREE.Vector3();
    for (const m of meshes) c.add(new THREE.Vector3().setFromMatrixPosition(m.userData.base));
    return c.multiplyScalar(1 / meshes.length);
  }
  function startAct(meshes, anim) {
    const c = centreOf(meshes), a = { meshes, anim, start: performance.now() + 1100, c };
    if (anim.kind === 'slide') {
      const from = copy.byObject.get(anim.away);
      a.dir = from ? c.clone().sub(centreOf(from)).setZ(0).normalize() : new THREE.Vector3(1, 0, 0);
      const arrow = new THREE.ArrowHelper(a.dir, c.clone().addScaledVector(a.dir, 0.3).setZ(c.z + 0.1), 0.5, 0xf0a93c, 0.16, 0.1);
      arrow.line.material.depthTest = arrow.cone.material.depthTest = false; arrow.renderOrder = 5;
      copy.root.add(arrow); extras.push(arrow); a.arrow = arrow;
      if (anim.want && anim.m < anim.want * 0.5) {     // where it was asked to go
        const ghost = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.22, depthWrite: false });
        const shift = new THREE.Matrix4().makeTranslation(a.dir.x * anim.want, a.dir.y * anim.want, 0);
        for (const m of meshes) { const g = new THREE.Mesh(m.geometry, ghost); g.matrixAutoUpdate = false; g.matrix.multiplyMatrices(shift, m.userData.base); copy.root.add(g); extras.push(g); }
      }
    }
    act = a;
  }
  function stepAct(now) {
    const a = act, t = (now - a.start) / 1000, T = new THREE.Matrix4();
    // 0-1.5 s out, a pause, then back if the task pushes it back
    const out = ease(t / 1.5), back = a.anim.back ? ease((t - 2.3) / 1.3) : 0, k = out - back;
    if (a.anim.kind === 'slide') {
      T.makeTranslation(a.dir.x * a.anim.m * k, a.dir.y * a.anim.m * k, 0);
      if (a.arrow) { a.arrow.visible = t > -0.6 && t < 1.7; a.arrow.setLength(0.38 + 0.12 * Math.abs(Math.sin(now / 160)), 0.16, 0.1); }
    } else {
      T.makeTranslation(a.c.x, a.c.y, 0).multiply(new THREE.Matrix4().makeRotationZ(a.anim.rad * k)).multiply(new THREE.Matrix4().makeTranslation(-a.c.x, -a.c.y, 0));
    }
    for (const m of a.meshes) m.matrix.multiplyMatrices(T, m.userData.base);
    copy.root.updateMatrixWorld(true);
    if (box) { box.box.makeEmpty(); for (const m of a.meshes) box.box.expandByObject(m); box.box.expandByScalar(0.03); }
    if (t > (a.anim.back ? 4.2 : 2.2)) { if (a.arrow) a.arrow.visible = false; if (a.anim.loop !== false) a.start = now + 900; }
  }
  /* A recording of a test run in this world: each moving body's pose over time, played on the world's own shapes. */
  const pose = (p, q) => new THREE.Matrix4().compose(new THREE.Vector3(p[0], p[1], p[2]), new THREE.Quaternion(q[1], q[2], q[3], q[0]), new THREE.Vector3(1, 1, 1));
  function stopRec() {
    if (!rec) return;
    for (const t of rec.tracks) for (const m of t.meshes) m.matrix.copy(m.userData.base);
    rec = null;
  }
  function startRec(data, lead) {
    stopRec();
    const tracks = [];
    for (const [name, b] of Object.entries(data.bodies)) {
      const meshes = copy.byBody.get(name);
      if (!meshes) continue;                       // the tester's own hand, or a body with no visible shape
      tracks.push({ meshes, restInv: pose(b.rest, b.rest.slice(3)).invert(), p: b.p, q: b.q });
    }
    rec = { tracks, times: data.times, duration: data.duration, start: performance.now() + (lead == null ? 900 : lead), M: new THREE.Matrix4() };
  }
  function stepRec(now) {
    const r = rec, span = r.duration + 1.6;       // hold the last pose, then play again
    const t = Math.max(0, Math.min(((now - r.start) / 1000) % span, r.duration));
    let i = Math.min(r.times.length - 1, Math.floor(t / r.duration * (r.times.length - 1)));
    while (i < r.times.length - 1 && r.times[i + 1] <= t) i++;
    while (i > 0 && r.times[i] > t) i--;
    for (const k of r.tracks) {
      const j = Math.min(i, k.p.length - 1);
      r.M.copy(pose(k.p[j], k.q[j])).multiply(k.restInv);
      for (const m of k.meshes) m.matrix.multiplyMatrices(r.M, m.userData.base);
    }
    copy.root.updateMatrixWorld(true);
    if (box && lit.length) { box.box.makeEmpty(); for (const m of lit) box.box.expandByObject(m); box.box.expandByScalar(0.03); }
    return t;
  }
  function clearLit() {
    stopRec();
    if (act) { for (const m of act.meshes) m.matrix.copy(m.userData.base); act = null; }
    for (const e of extras) { e.removeFromParent(); if (e.isMesh) e.material.dispose(); else e.dispose?.(); }
    extras = [];
    for (const m of lit) m.material.emissive.set(0);
    lit = [];
    if (box) { scene.remove(box); box.geometry.dispose(); box.material.dispose(); box = null; }
  }
  return {
    show(data, cam) {
      clearLit(); centre = null;
      if (copy) { scene.remove(copy.root); copy.root.traverse(o => { if (o.isMesh) o.material.dispose(); }); }
      copy = buildCopy(data); scene.add(copy.root);
      // The public input camera, moved from the copy's z-up frame into three.js's y-up frame.
      let pos, target;
      if (cam) {
        const m = new THREE.Matrix4().set(...cam.m.flat()).premultiply(UP);
        pos = new THREE.Vector3().setFromMatrixPosition(m);
        target = new THREE.Vector3(...cam.target).applyMatrix4(UP);
        camera.fov = cam.fov;
      } else {
        // No input camera for this scene: a three-quarter view fitted to the furniture (the floor plane is left out).
        const b = new THREE.Box3(), p = new THREE.Vector3();
        for (const g of data.geoms) if (g.t !== 'plane') b.expandByPoint(p.set(g.p[0], g.p[1], g.p[2]).applyMatrix4(UP));
        const c = b.getCenter(new THREE.Vector3()), size = b.getSize(new THREE.Vector3()), span = Math.max(size.x, size.z, 2);
        target = new THREE.Vector3(c.x, b.min.y + Math.min(size.y * 0.25, 0.9), c.z); centre = c.clone();
        pos = new THREE.Vector3(c.x + span * 0.62, b.min.y + span * 0.7, c.z + span * 0.8);
        camera.fov = 42;
      }
      camera.updateProjectionMatrix();
      home = { pos, target }; go(pos.clone(), target.clone(), false);
    },
    home() { clearLit(); if (home) go(home.pos.clone(), home.target.clone(), true); },
    play(data, lead) { if (copy) startRec(data, lead); },
    lock(on) { controls.enabled = !on; },
    look(eye, target, fov) { copy.root.traverse(o => { if (o.userData.shell) o.visible = false; }); camera.fov = fov; camera.updateProjectionMatrix(); home = { pos: new THREE.Vector3(...eye), target: new THREE.Vector3(...target) }; go(home.pos.clone(), home.target.clone(), false); },
    project(name) {
      const meshes = copy?.byObject.get(name); if (!meshes) return null;
      copy.root.updateMatrixWorld(true); camera.updateMatrixWorld(true);
      const c = new THREE.Box3(); for (const m of meshes) c.expandByObject(m);
      const v = c.getCenter(new THREE.Vector3()).project(camera);
      return [(v.x + 1) / 2, (1 - v.y) / 2];
    },
    xray(name, on) {
      for (const m of copy?.byObject.get(name) || []) { m.material.wireframe = on; m.material.needsUpdate = true; }
    },
    point(name, ok, anim, near) {
      clearLit();
      const meshes = copy?.byObject.get(name);
      if (!meshes) return;
      copy.root.updateMatrixWorld(true);
      const b = new THREE.Box3();
      for (const m of meshes) { m.material.emissive.set(ok ? 0x0e5e3f : 0x7a1d14); lit.push(m); b.expandByObject(m); }
      box = new THREE.Box3Helper(b.clone().expandByScalar(0.03), ok ? 0x22c58b : 0xff5a4a);
      box.material.depthTest = false; box.material.transparent = true; scene.add(box);
      const c = b.getCenter(new THREE.Vector3()), size = b.getSize(new THREE.Vector3()).length();
      // Stand between the middle of the room and the object, a little above it, so walls never get in the way.
      const dir = (centre ? centre.clone().sub(c).setY(0) : camera.position.clone().sub(c));
      if (dir.length() < 0.3) dir.copy(camera.position).sub(c).setY(0);
      dir.normalize(); dir.y = centre ? 0.55 : Math.max(dir.y, 0.35); dir.normalize();
      go(c.clone().addScaledVector(dir, Math.max(near || (anim ? 2.3 : 1.5), size * 1.25)), c, true);
      if (anim) startAct(meshes, anim);
    }
  };
}

/* The two-pane player used by "Looks against works" and "Caught in the test": the reference world on the left and one
   submitted world on the right, from the same camera, playing the same recorded test together, on a loop. */
function makePlayer(figT, figW) {
  const worlds = new Map(), recs = new Map(), scenes = new Map();
  const get = (map, url) => { if (!map.has(url)) map.set(url, fetch(url).then(r => r.json())); return map.get(url); };
  let viewer = null, stage = null, catalog = null, loaded = null, timer = 0, want = 0;
  return async function show(d) {
    const mine = ++want;
    clearInterval(timer);
    $('figcaption', figT).textContent = d.truth; const cap = $('figcaption', figW); cap.textContent = d.agent; cap.className = d.ok ? 'ok' : 'no';
    if (!catalog) catalog = (await (await fetch('scenes/assets/rooms.json')).json()).rooms;
    if (mine !== want) return;
    if (!viewer) { viewer = createViewer($('canvas', figT), figT, s => { if (s.replay && s.replay.duration) figT.classList.add('live'); }, false); stage = createStage($('canvas', figW), figW); }
    if (loaded !== d.scene) {
      loaded = null; figT.classList.remove('live');
      if (!await viewer.load(catalog.find(r => r.id === d.scene)) || mine !== want) return;
      loaded = d.scene;
    }
    const [replays, world, rec] = await Promise.all([get(scenes, `scenes/assets/rooms/${d.scene}/replays.json`), get(worlds, `data/copies/${d.world}.json`), get(recs, `data/recordings/${d.world}--${d.task}.json`)]);
    if (mine !== want) return;
    const meta = { ...replays[d.task] };
    // stand a little further back than the recording's own close-up, so the surroundings are in view
    meta.eye = meta.eye.map((v, k) => meta.target[k] + (v - meta.target[k]) * (d.back || 1.5));
    stage.show(world, null); stage.look(meta.eye, meta.target, 39); figW.classList.add('live');
    await viewer.playReplay(meta);
    if (mine !== want) return;
    stage.play(rec, 700);
    const period = (Math.max((meta.frames - 1) / meta.fps, rec.duration) + 1.8) * 1000;
    timer = setInterval(() => { if (document.hidden || mine !== want) return; viewer.seekReplay(0); viewer.toggleReplay(); stage.play(rec, 0); }, period);
  };
}

/* "Caught in the test": pick a world; its tests are a row of pass and fail marks; the first plays by itself. */
function cases() {
  const root = $('#cases'), list = window.SB_CASES;
  if (!root || !list) return;
  const tabs = $('.casetabs', root), chips = $('.testchips', root);
  const show = makePlayer($('.dtrue', root), $('.dworld', root));
  let current = list[0], started = false;
  tabs.innerHTML = list.map((c, i) => `<button type="button" data-i="${i}" aria-pressed="${i === 0}"><small>${esc(c.place)}</small>${esc(c.agent)}</button>`).join('');
  function play(k) {
    chips.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.k === k));
    const check = current.checks[k];
    $('.verdictline', root).textContent = check.verdict;
    if (started) show({ scene: current.scene, task: check.replay, world: current.copy, back: 1.5, truth: check.short, agent: check.read, ok: check.ok });
  }
  function open(i) {
    current = list[i];
    tabs.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-pressed', j === i));
    chips.innerHTML = current.checks.map((k, j) => k.rec ? `<button type="button" data-k="${j}" class="${k.ok ? 'ok' : 'no'}" aria-pressed="false"><i>${k.ok ? '✓' : '✕'}</i>${esc(k.short)}</button>` : '').join('');
    play(current.checks.findIndex(k => k.rec));
  }
  tabs.addEventListener('click', e => { const b = e.target.closest('button'); if (b) open(+b.dataset.i); });
  chips.addEventListener('click', e => { const b = e.target.closest('button'); if (b) play(+b.dataset.k); });
  open(0);
  new IntersectionObserver((es, io) => { if (es[0].isIntersecting) { io.disconnect(); started = true; play(current.checks.findIndex(k => k.rec)); } }, { rootMargin: '200px' }).observe(root);
}

/* "Looks against works": one test, twice. The reference world and one submitted world, from the same camera, playing the
   same recorded test together. A row of examples to pick from; the first starts when the section scrolls into view. */
const DUELS = [
  { name: 'The stock pot', head: 'carries', scene: 'f6-commercial-kitchen-001', task: 'stock_pot_to_burner', world: 'kitchen-gemini', back: 1.7,
    truth: 'The pot is taken off the shelf and set on the burner.', agent: 'The pot is not pulled clear of the shelf. It ends 3.5 m from the burner.', who: 'Commercial kitchen · Gemini 4 · appearance 67.9' },
  { name: 'The supply drawer', head: 'holds', scene: 'delivery-room-001', task: 'access_warmer_storage', world: 'delivery-astra', back: 1.6,
    truth: 'The warmer rises, the drawer opens with its boxes, and closes.', agent: 'The warmer never rises. None of the four steps is completed.', who: 'Delivery room · GPT-6 Astra, max effort · appearance 85.7' },
  { name: 'The mail folder', head: 'comes out', scene: 'school-front-office-001', task: 'distribute_mail_folder', world: 'school-kimi', back: 1.6,
    truth: 'The folder is pulled from its cubby and laid on the table.', agent: 'The folder does not come out. It ends 2.3 m from the table spot.', who: 'School front office · Kimi K3.1 Pro · appearance 65.2' },
  { name: 'The culture bottle', head: 'fits', scene: 'biology-lab', task: 'bottle_to_fridge', world: 'biology-astra', back: 1.6,
    truth: 'The bottle goes from the cabinet onto a fridge shelf.', agent: 'The bottle jams on the way: the grasp pulls with 51 N.', who: 'Biology lab · GPT-6 Astra, max effort · appearance 87.7' },
  { name: 'The laptop', head: 'closes', scene: 'humanoid-test-lab-001', task: 'probe_laptop_hinge', world: 'humanoid-astra', back: 1.6,
    truth: 'The lid is set partly closed, holds there, then shuts.', agent: 'The lid cannot be set partly closed. All four steps fail.', who: 'Humanoid test lab · GPT-6 Astra · appearance 89.8' },
  { name: 'The monitor', head: 'tilts', scene: 'delivery-room-001', task: 'prepare_monitor_cart', world: 'delivery-astra', back: 1.6,
    truth: 'The drawer opens, the display tilts toward you, the drawer closes.', agent: 'The drawer opens and closes. The display does not tilt.', who: 'Delivery room · GPT-6 Astra, max effort · appearance 85.7' },
  { name: 'The stapler', head: 'opens', scene: 'school-front-office-001', task: 'open_staple_magazine', world: 'school-astra', back: 1.6,
    truth: 'The head lifts and the staple magazine slides out, then stows.', agent: 'The head does not lift. The magazine moves 1 mm.', who: 'School front office · GPT-6 Astra, max effort · appearance 81.4' },
  { name: 'The toy car', head: 'rolls', scene: 'childrens-room-001', task: 'push_road_car', world: 'childrens-opus', back: 2.1,
    truth: 'Pushed once, the car rolls across the rug.', agent: 'Same push. The car does not move: 0.0 cm.', who: "Children's room · Claude Opus 5.5, max effort · appearance 84.0" }
];
function duel() {
  const root = $('#duel');
  if (!root) return;
  const tabs = $('.dueltabs', root), headWord = $('#duelword');
  const show = makePlayer($('.dtrue', root), $('.dworld', root));
  let started = false, at = 0;
  tabs.innerHTML = DUELS.map((d, i) => `<button type="button" data-i="${i}" aria-pressed="${i === 0}">${esc(d.name)}</button>`).join('');
  function pick(i) {
    at = i; const d = DUELS[i];
    $('.duelwho', root).textContent = d.who; if (headWord) headWord.textContent = d.head;
    tabs.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-pressed', j === i));
    if (started) show(d);
  }
  tabs.addEventListener('click', e => { const b = e.target.closest('button'); if (b) pick(+b.dataset.i); });
  pick(0);
  new IntersectionObserver((es, io) => { if (es[0].isIntersecting) { io.disconnect(); started = true; show(DUELS[at]); } }, { rootMargin: '300px' }).observe(root);
}

cases();
duel();
