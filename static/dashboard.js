(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var latest = null, shown = null, paused = false, highlight = null, view = "smile";
  var arbState = { sortDesc: true }, retry = 0, retryTimer = null, ws = null;

  try {
    var saved = localStorage.getItem("theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (e) { /* storage blocked; follow the system theme */ }

  function banner(text, action) {
    var el = $("banner");
    el.hidden = !text;
    el.textContent = text || "";
    if (action) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = action.label;
      b.addEventListener("click", action.run);
      el.appendChild(b);
    }
  }

  function render(p) {
    shown = p;
    $("waiting").hidden = true;
    $("views").hidden = false;
    $("source").textContent = p.source;
    $("spot").textContent = p.spot.toFixed(2);
    $("tick").textContent = p.tick;
    var t = VS.takeaways(p);
    $("smile-takeaway").textContent = t.smile;
    $("term-takeaway").textContent = t.term;
    $("skew-takeaway").textContent = t.skew;
    $("surface-takeaway").textContent = t.surface || "";
    if (view === "smile") VS.renderSmile($("smile"), p, highlight);
    else VS.renderSurface($("surface"), p);
    VS.renderFailed($("failed"), p);
    VS.renderTerm($("term"), p);
    VS.renderSkew($("skew"), p);
    VS.renderArbTable($("arb"), p, arbState, function (v) {
      highlight = VS.highlightFor(p, v);
      if (view !== "smile") selectView("smile");
      VS.renderSmile($("smile"), p, highlight);
    });
  }

  function selectView(name) {
    view = name;
    ["smile", "surface"].forEach(function (n) {
      $("view-" + n).hidden = n !== name;
      $("tab-" + n).setAttribute("aria-selected", String(n === name));
    });
    if (shown) render(shown);
  }

  function connect() {
    clearTimeout(retryTimer);
    var scheme = location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(scheme + location.host + "/ws");
    ws.onopen = function () { retry = 0; banner(""); };
    ws.onmessage = function (event) {
      var msg = JSON.parse(event.data);
      if (msg.error) {
        banner("Live source error: " + msg.error + ". Showing the last good snapshot (stale).");
        $("views").classList.add("stale");
        return;
      }
      $("views").classList.remove("stale");
      latest = msg;
      if (!paused) render(msg);
    };
    ws.onclose = function () {
      var wait = Math.min(30, Math.pow(2, retry++));
      banner("Connection closed. Retrying in " + wait + " s.", { label: "Reconnect now", run: connect });
      retryTimer = setTimeout(connect, wait * 1000);
    };
  }

  $("pause").addEventListener("click", function () {
    paused = !paused;
    this.setAttribute("aria-pressed", String(paused));
    this.textContent = paused ? "Resume" : "Pause";
    if (!paused && latest) render(latest);
  });
  $("theme").addEventListener("click", function () {
    var dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    var next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* not saved; still applied */ }
    if (shown) render(shown);
  });
  $("tab-smile").addEventListener("click", function () { selectView("smile"); });
  $("tab-surface").addEventListener("click", function () { selectView("surface"); });
  connect();
})();
