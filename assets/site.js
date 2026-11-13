/* 3DSceneBench site. Plain JS, no build step. Data comes from data/*.js (written by build.py).
   The 3D scene gallery lives in scene.js. */
(function () {
  'use strict';
  var C = { A: 'var(--app)', G: 'var(--geo)', F: 'var(--fun)', ink: 'var(--ink)' };
  var $ = function (s, el) { return (el || document).querySelector(s); };
  var $$ = function (s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var fmt = function (v, d) { return v == null ? '–' : Number(v).toFixed(d == null ? 1 : d); };
  var pic = function (n, k) { return 'img/rooms/' + n + '/input_' + k + '.jpg'; };
  var still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var avg = function (xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; };

  /* Theme: follows the system until the visitor picks one. */
  function theme() {
    var saved = null;
    try { saved = localStorage.getItem('sb-theme'); } catch (e) {}
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    var b = $('.theme');
    if (b) b.addEventListener('click', function () {
      var cur = document.documentElement.getAttribute('data-theme') || 'light';
      var next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('sb-theme', next); } catch (e) {}
    });
  }

  /* Things that happen once, when an element scrolls into view. */
  function onSeen(el, fn, threshold) {
    if (!el) return;
    if (!('IntersectionObserver' in window)) return fn(el);
    var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); fn(e.target); } }); }, { threshold: threshold || 0.2 });
    io.observe(el);
  }
  function countUp(el, to, d, prefix, suffix) {
    prefix = prefix || ''; suffix = suffix || '';
    if (still) { el.textContent = prefix + fmt(to, d) + suffix; return; }
    var start = performance.now(), from = parseFloat(el.dataset.shown || 0) || 0;
    el.dataset.shown = to;
    (function step(now) {
      var t = Math.min((now - start) / 900, 1), e = 1 - Math.pow(1 - t, 3);
      el.textContent = prefix + fmt(from + (to - from) * e, d) + suffix;
      if (t < 1) requestAnimationFrame(step);
    })(start);
  }

  function chrome() {
    var bar = $('.progress i'), top = $('.top');
    var onScroll = function () {
      var h = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.transform = 'scaleX(' + (h > 0 ? scrollY / h : 0) + ')';
      if (top) top.classList.toggle('scrolled', scrollY > 40);
    };
    addEventListener('scroll', onScroll, { passive: true }); onScroll();
    $$('.reveal').forEach(function (el) { onSeen(el, function (x) { x.classList.add('in'); }, 0.12); });
    setTimeout(function () { document.body.classList.add('loaded'); }, 60);

    var dialog = $('#film');
    if (dialog) {
      var v = $('video', dialog);
      var close = function () { v.pause(); v.removeAttribute('src'); v.load(); if (dialog.open) dialog.close(); };
      $('button', dialog).addEventListener('click', close);
      dialog.addEventListener('cancel', function (e) { e.preventDefault(); close(); });
      dialog.addEventListener('click', function (e) { if (e.target === dialog) close(); });
      $$('[data-film]').forEach(function (b) {
        b.addEventListener('click', function () { $('h3', dialog).textContent = b.dataset.title || ''; v.src = b.dataset.film; dialog.showModal(); v.play().catch(function () {}); });
      });
    }
  }

  /* ---------- Results table (leaderboard page) ---------- */
  var MAIN = [
    { key: 'rank', label: '#', d: 0, asc: true },
    { key: 'agent', label: 'Agent', text: true },
    { key: 'F', label: 'Functionality', cls: 'f', bar: C.F },
    { key: 'A', label: 'Appearance', cls: 'a', bar: C.A },
    { key: 'G', label: 'Geometry', cls: 'g', bar: C.G },
    { key: 'tasks_full_pct', label: 'Tasks in full %' },
    { key: 'valid_pct', label: 'Valid %', d: 0 },
    { key: 'minutes', label: 'Min / build', d: 0, lowBest: true },
    { key: 'usd_list', label: 'USD / build', lowBest: true, cost: true }
  ];
  var PARTS = [
    { key: 'rank', label: '#', d: 0, asc: true },
    { key: 'agent', label: 'Agent', text: true },
    { key: 'balance', label: 'Balance', cls: 'f' },
    { key: 'response', label: 'Response', cls: 'f' },
    { key: 'affordance', label: 'Affordance', cls: 'f' },
    { key: 'interaction', label: 'Interaction', cls: 'f' },
    { key: 'siglip', label: 'SigLIP', cls: 'a', d: 2 },
    { key: 'dino', label: 'DINO', cls: 'a', d: 2 },
    { key: 'inventory', label: 'Inventory', cls: 'g' },
    { key: 'shape', label: 'Shape', cls: 'g' },
    { key: 'pose_size', label: 'Pose and size', cls: 'g' },
    { key: 'relations', label: 'Relations', cls: 'g' },
    { key: 'placement', label: 'Placement', cls: 'g' }
  ];
  function board(el, rows, cols) {
    var sort = { key: 'rank', dir: 1 };   // the paper's order: by functionality
    function draw() {
      var data = rows.slice().sort(function (a, b) {
        var x = a[sort.key], y = b[sort.key];
        if (typeof x === 'string') return sort.dir * x.localeCompare(y);
        return sort.dir * ((x == null ? -1e9 : x) - (y == null ? -1e9 : y));
      });
      var best = {};
      cols.forEach(function (c) {
        if (c.text || c.key === 'rank' || c.lowBest) return;
        best[c.key] = Math.max.apply(null, rows.map(function (r) { return r[c.key]; }).filter(function (v) { return v != null; }));
      });
      var h = '<table class="board"><thead><tr>' + cols.map(function (c) {
        var s = sort.key === c.key ? ' aria-sort="' + (sort.dir < 0 ? 'descending' : 'ascending') + '"' : '';
        return '<th scope="col" data-key="' + c.key + '" class="' + (c.text ? 'l ' : '') + (c.cls || '') + '"' + s + '>' + c.label + '</th>';
      }).join('') + '</tr></thead><tbody>';
      data.forEach(function (r) {
        h += '<tr>' + cols.map(function (c) {
          var v = r[c.key];
          if (c.key === 'agent') return '<td class="l agent">' + esc(v) + '<span class="tag">' + esc(r.effort) + '</span></td>';
          if (c.text) return '<td class="l">' + esc(v) + '</td>';
          var bar = c.bar && v != null ? '<span class="cellbar"><i style="width:' + Math.max(0, Math.min(100, v)) + '%;background:' + c.bar + '"></i></span>' : '';
          return '<td class="num' + (best[c.key] != null && v === best[c.key] ? ' best' : '') + '">' + bar + (c.cost ? (v < 1 ? v.toFixed(2) : v < 10 ? v.toFixed(1) : v.toFixed(0)) : fmt(v, c.d)) + '</td>';
        }).join('') + '</tr>';
      });
      el.innerHTML = h + '</tbody></table>';
      $$('th[data-key]', el).forEach(function (th) {
        th.addEventListener('click', function () {
          var k = th.getAttribute('data-key'), col = cols.filter(function (c) { return c.key === k; })[0];
          if (sort.key === k) sort.dir = -sort.dir; else sort = { key: k, dir: (col.text || col.asc || col.lowBest) ? 1 : -1 };
          draw();
        });
      });
    }
    draw();
  }
  function leaderboard() {
    var R = window.SB_RESULTS;
    if (!R) return;
    var el = $('#board'), view = 'main';
    var draw = function () { board(el, R.table, view === 'main' ? MAIN : PARTS); };
    $$('.controls .seg button').forEach(function (b) {
      b.addEventListener('click', function () {
        view = b.getAttribute('data-view');
        $$('.controls .seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); });
        draw();
      });
    });
    draw();
    var s = $('#board-source');
    if (s) s.textContent = R.builds_l0 + ' builds at L0, ' + R.table.length + ' configurations. Data generated ' + R.generated + '.';
  }

  /* ---------- Landing page ---------- */
  function ticker(R, rooms) {
    var el = $('#ticker');
    if (!el) return;
    var items = [rooms.length + ' scenes', rooms.reduce(function (n, r) { return n + (r.tasks || 0); }, 0) + ' tasks', '8 pictures in', '1 working world out', 'MuJoCo 3.14', 'Genesis 1.4.3', 'Appearance', 'Geometry', 'Functionality'];
    if (R) items.splice(2, 0, R.table.length + ' agent configurations', R.builds + ' graded builds');
    var h = items.map(function (t) { return '<span>' + esc(t) + '</span><i>◆</i>'; }).join('');
    el.innerHTML = h + h;
  }
  // Short names for the scene strip, written from each scene's public description.
  var NAMES = {"001": "Gym", "002": "Breakfast shop", "003": "Open-plan office", "004": "Operating room", "005": "Living room", "006": "Family bathroom", "007": "Recording studio", "008": "Assisted-living bathroom", "009": "Laundry room", "010": "Hotel bedroom", "011": "Convenience store", "012": "Cozy bedroom", "013": "Housekeeping stockroom", "014": "Pump overhaul hall", "015": "Delivery room", "016": "Dining room", "017": "Biology lab", "018": "Open-plan living area", "019": "Utility-skid training room", "020": "Commercial kitchen", "021": "Parcel sorting hall", "022": "Outdoor equipment shop", "023": "Furniture workshop", "024": "Electronics workshop", "025": "Datacenter service room", "026": "Specialty café", "027": "Rehabilitation hall", "028": "Auto repair workshop", "029": "Supermarket", "030": "Hospital patient room", "031": "Udon restaurant", "032": "School front office", "033": "Machining cell", "034": "Chemistry teaching lab", "035": "Car assembly hall", "036": "Art studio", "037": "Children's playroom", "038": "Carpentry workshop", "039": "Home kitchen", "040": "Bakery café", "041": "Robotics test lab", "042": "Diner", "043": "Bedroom", "044": "Public library", "045": "Electronics lab", "046": "Optics lab", "047": "Family garage", "048": "Study", "049": "Packing workshop"};
  function wall(rooms) {
    var half = Math.ceil(rooms.length / 2);
    [['#wall-a', rooms.slice(0, half)], ['#wall-b', rooms.slice(half)]].forEach(function (x) {
      var el = $(x[0]);
      if (!el) return;
      var h = x[1].map(function (r) { return '<figure title="' + esc(r.description) + '"><img src="' + pic(r.n, 0) + '" alt="Scene ' + r.n + '" loading="lazy"><figcaption><b>' + r.n + '</b> ' + esc(NAMES[r.n] || '') + '</figcaption></figure>'; }).join('');
      el.innerHTML = h + h;
    });
  }
  function steps(R, rooms) {
    var fan = $('#fan');
    if (fan && rooms.length) fan.innerHTML = [0, 1, 2, 3, 4, 5, 6, 7].map(function (k) {
      return '<img src="' + pic(rooms[0].n, k) + '" alt="Input picture ' + (k + 1) + '" style="--i:' + (k - 3.5) + '" loading="lazy">';
    }).join('');

    var typed = $('#typed');
    if (typed) {
      var code = '<body name="cup__0" pos="1.3 1.35 0.74">\n  <freejoint/>\n  <geom type="mesh" mass="0.3"/>\n</body>\n<body name="cabinet__0">\n  <body name="drawer">\n    <joint type="slide"\n      axis="1 0 0" range="-0.4 0"/>\n  </body>\n</body>';
      if (still) typed.textContent = code;
      else onSeen(typed, function () {
        var i = 0;
        (function type() { typed.textContent = code.slice(0, i); i += 2; if (i <= code.length + 1) setTimeout(type, 16); else typed.classList.add('done'); })();
      }, 0.5);
    }
    if (R) {
      var best = {}; ['A', 'G', 'F'].forEach(function (k) { best[k] = Math.max.apply(null, R.table.map(function (r) { return r[k]; })); });
      onSeen($('.dials'), function () {
        $$('[data-dial]').forEach(function (c) { c.style.strokeDashoffset = 201.06 * (1 - best[c.dataset.dial] / 100); });
        $$('[data-count]').forEach(function (b) { countUp(b, best[b.dataset.count], 1); });
      }, 0.5);
    }
  }
  // The film under the introduction: a still with one button; a click plays it in place.
  function demo() {
    var box = $('.demo'); if (!box) return;
    var v = $('video', box), c = $('.democover', box);
    c.addEventListener('click', function () { box.classList.add('on'); v.controls = true; v.play().catch(function () {}); });
  }

  function reel() {
    var vids = $$('.hero video.bg'), box = $('.hero .reel');
    if (vids.length < 2 || !box || still) return;
    var F = 'scenes/assets/films/';
    var list = [
      { src: 'media/hero.mp4', len: 7, name: 'Living room with pool' },
      { src: F + 'biology-lab.mp4', len: 6, name: 'Biology lab' },
      { src: F + 'f6-commercial-kitchen-001.mp4', len: 6, name: 'Commercial kitchen' },
      { src: F + 'auto-repair-001.mp4', len: 6, name: 'Auto repair workshop' },
      { src: F + 'childrens-room-001.mp4', len: 5, name: "Children's room" },
      { src: F + 'rehab-center-001.mp4', len: 6, name: 'Rehab center' }
    ];
    var name = $('.reelname', box), dots = $('.reeldots', box), i = 0, front = 0, timer, fails = 0;
    dots.innerHTML = list.map(function (c) { return '<button type="button" aria-label="' + esc(c.name) + '"></button>'; }).join('');
    var btn = $$('button', dots);
    function mark() {
      name.textContent = list[i].name;
      btn.forEach(function (b, k) { b.className = k < i ? 'done' : ''; b.style.setProperty('--len', list[k].len + 's'); });
      void dots.offsetWidth; btn[i].className = 'on';
      clearTimeout(timer); timer = setTimeout(function () { show(i + 1); }, list[i].len * 1000);
    }
    function show(k) {
      clearTimeout(timer);
      if (document.hidden) { timer = setTimeout(function () { show(k); }, 2000); return; }
      k = (k + list.length) % list.length;
      var c = list[k], v = vids[1 - front], old = vids[front];
      v.oncanplay = function () {
        v.oncanplay = v.onerror = null; fails = 0;
        var p = v.play(); if (p && p.catch) p.catch(function () {});
        old.classList.remove('on'); v.classList.add('on'); front = 1 - front; i = k; mark();
        setTimeout(function () { old.pause(); }, 1400);
      };
      v.onerror = function () { v.oncanplay = v.onerror = null; if (++fails < list.length) show(k + 1); };
      v.preload = 'auto'; v.src = c.src; v.load();
    }
    box.hidden = false; mark();
    dots.addEventListener('click', function (e) { var k = btn.indexOf(e.target.closest('button')); if (k >= 0 && k !== i) show(k); });
  }
  function mugs() {
    var lab = $('.mugs .lab'), drop = $('#drop'), xray = $('#xray');
    if (!lab || !drop) return;
    drop.addEventListener('click', function () {
      lab.classList.remove('drop'); void lab.offsetWidth; lab.classList.add('drop');
      drop.textContent = 'Drop it again';
    });
    xray.addEventListener('click', function () {
      var on = lab.classList.toggle('xrayed'); xray.setAttribute('aria-pressed', String(on));
    });
    onSeen(lab, function () { if (!still) setTimeout(function () { lab.classList.add('drop'); }, 500); }, 0.6);
  }
  function race(R) {
    var lanes = $('#lanes');
    if (!lanes || !R) return;
    var T = R.table, metric = 'reward';
    var META = {
      reward: ['Reward: the mean of appearance, geometry and functionality.', C.ink],
      A: ['Appearance: how the world looks from 16 cameras the agent never saw.', C.A],
      G: ['Geometry: the right objects, the right size, in the right places.', C.G],
      F: ['Functionality: how the world behaves when pushed, opened and used by a robot.', C.F],
      tasks_full_pct: ['Share of a scene\'s own tasks that the world completes in full, in percent.', C.F]
    };
    lanes.innerHTML = T.map(function (r, i) {
      return '<li data-i="' + i + '"><span class="pos"></span><span class="who">' + esc(r.agent) + '<small>' + esc(r.effort) + ' · ' + esc(r.harness) + '</small></span><span class="track"><i></i></span><b class="val">0</b></li>';
    }).join('');
    var rows = $$('li', lanes);
    function draw(animate) {
      var order = T.map(function (r, i) { return i; }).sort(function (a, b) { return T[b][metric] - T[a][metric]; });
      var first = {};
      rows.forEach(function (li) { first[li.dataset.i] = li.getBoundingClientRect().top; });
      order.forEach(function (i, pos) {
        var li = rows[i];
        lanes.appendChild(li);
        $('.pos', li).textContent = pos + 1;
        var bar = $('.track i', li);
        bar.style.background = META[metric][1];
        bar.style.width = Math.max(1, T[i][metric]) + '%';
        if (animate) countUp($('.val', li), T[i][metric], 1); else $('.val', li).textContent = fmt(T[i][metric]);
      });
      if (animate && !still) rows.forEach(function (li) {
        var dy = first[li.dataset.i] - li.getBoundingClientRect().top;
        if (!dy) return;
        li.style.transition = 'none'; li.style.transform = 'translateY(' + dy + 'px)';
        requestAnimationFrame(function () { requestAnimationFrame(function () { li.style.transition = ''; li.style.transform = ''; }); });
      });
      $('#racenote').textContent = META[metric][0];
    }
    $$('.race .seg button').forEach(function (b) {
      b.addEventListener('click', function () {
        metric = b.dataset.metric;
        $$('.race .seg button').forEach(function (x) { x.setAttribute('aria-pressed', x === b); });
        draw(true);
      });
    });
    rows.forEach(function (li) { $('.track i', li).style.width = '0%'; });
    onSeen(lanes, function () { draw(true); }, 0.25);
  }
  function findings(R) {
    if (!R) return;
    var T = R.table, by = function (k) { return T.slice().sort(function (a, b) { return b[k] - a[k]; })[0]; };
    var bA = by('A'), bF = by('F'), bFull = by('tasks_full_pct'), L = R.levels.slice(0, 4);
    var d = function (k) { return avg(L.map(function (r) { return r.L1[k] - r.L0[k]; })); };
    var sg = function (v) { return (v >= 0 ? '+' : '') + fmt(v); };
    var nums = { bestA: bA.A, bestF: bF.F, bestFull: bFull.tasks_full_pct, l1G: L.length ? d('G') : null };
    var set = function (id, t) { var e = document.getElementById(id); if (e) e.textContent = t; };
    set('f1', bA.agent + ', ' + bA.effort + ' effort. From cameras the agent never saw, the best worlds look close to the scene.');
    set('f2', bF.agent + ', ' + bF.effort + ' effort. Once the same worlds are pushed, opened and used, the score falls by almost half.');
    set('f3', bFull.agent + ', ' + bFull.effort + ' effort. Even the strongest agents finish about a quarter of a scene\'s tasks completely.');
    if (L.length) set('f4', 'Level L1 adds an inventory, measurements and depth. The four strongest configurations gain ' + sg(d('G')) + ' on geometry and ' + sg(d('F')) + ' on functionality; appearance moves ' + sg(d('A')) + '.');
    $$('[data-num]').forEach(function (b) {
      var v = nums[b.dataset.num];
      if (v == null) { b.closest('.finding').hidden = true; return; }
      onSeen(b, function () { countUp(b, v, 1, b.dataset.prefix, b.dataset.suffix); }, 0.6);
    });
    set('n-builds', R.builds); set('n-configs', T.length);
  }
  function landing() {
    var R = window.SB_RESULTS, rooms = window.SB_ROOMS || [];
    var set = function (id, v) { var e = document.getElementById(id); if (e) e.textContent = v; };
    set('n-rooms', rooms.length); set('n-tasks', rooms.reduce(function (n, r) { return n + (r.tasks || 0); }, 0));
    reel(); demo(); wall(rooms); steps(R, rooms); mugs(); leaderboard();
  }

  document.addEventListener('DOMContentLoaded', function () {
    theme(); chrome();
    var page = document.body.getAttribute('data-page');
    if (page === 'landing') landing();
    if (page === 'leaderboard') leaderboard();
  });
})();
