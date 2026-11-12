// Renders a snapshot payload (see analysis.py) into the smile, surface,
// term-structure, skew and arbitrage views. Used by the live page and the
// static report.
(function () {
  "use strict";

  // Okabe-Ito, one per expiry; dashes and marker shapes repeat the encoding.
  var COLORS = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"];
  var DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"];
  var SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x"];

  var state = { payload: null, selected: null, sortDesc: true, tab: "smile" };

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function pct(x, digits) {
    return x === null || x === undefined ? "n/a" : (x * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function el(id) {
    return document.getElementById(id);
  }

  function baseLayout(xTitle, yTitle) {
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: css("--text"), size: 12 },
      margin: { l: 56, r: 12, t: 8, b: 44 },
      xaxis: { title: { text: xTitle }, gridcolor: css("--grid"), zerolinecolor: css("--border") },
      yaxis: { title: { text: yTitle }, gridcolor: css("--grid"), zerolinecolor: css("--border") },
      legend: { orientation: "h", x: 0, xanchor: "left", y: -0.22, yanchor: "top",
        title: { text: "Click an expiry to toggle it", side: "top" } },
      hoverlabel: { bgcolor: css("--panel"), font: { color: css("--text") } },
    };
  }

  var CONFIG = { responsive: true, displaylogo: false };

  function smileQuotes(p, exp) {
    return p.quotes.filter(function (q) {
      return q.smile && q.expiry === exp.days;
    });
  }

  function curveIvAt(exp, k) {
    var s = exp.svi;
    if (!s) return null;
    var x = k - s.m;
    var w = s.a + s.b * (s.rho * x + Math.sqrt(x * x + s.s * s.s));
    return w > 0 ? Math.sqrt(w / exp.T) : null;
  }

  function quoteHover(q) {
    return (
      q.type + " K=" + q.strike.toFixed(2) + "<br>expiry " + q.expiry.toFixed(1) + "d" +
      "<br>price " + q.price.toFixed(4) +
      "<br>IV " + (q.iv === null ? "none (" + q.reason + ")" : pct(q.iv, 2)) +
      "<br>iterations " + q.iterations + "<extra></extra>"
    );
  }

  function renderSmile(p) {
    var traces = [];
    p.expiries.forEach(function (exp, i) {
      var group = exp.days.toFixed(0) + "d";
      var color = COLORS[i % COLORS.length];
      if (exp.curve) {
        traces.push({
          type: "scatter", mode: "lines", name: group + " SVI fit", legendgroup: group,
          x: exp.curve.k, y: exp.curve.iv.map(function (v) { return v * 100; }),
          line: { color: color, dash: DASHES[i % DASHES.length], width: 2 },
          hovertemplate: group + " fit<br>k=%{x:.4f}<br>IV %{y:.2f}%<extra></extra>",
        });
      }
      var qs = smileQuotes(p, exp);
      var ok = qs.filter(function (q) { return q.iv !== null; });
      var bad = qs.filter(function (q) { return q.iv === null && curveIvAt(exp, q.k) !== null; });
      traces.push({
        type: "scatter", mode: "markers", name: group + " quotes", legendgroup: group,
        x: ok.map(function (q) { return q.k; }), y: ok.map(function (q) { return q.iv * 100; }),
        marker: { color: color, symbol: SYMBOLS[i % SYMBOLS.length], size: 7 },
        hovertemplate: ok.map(quoteHover),
      });
      if (bad.length) {
        traces.push({
          type: "scatter", mode: "markers", name: group + " failed solves", legendgroup: group,
          showlegend: false,
          x: bad.map(function (q) { return q.k; }),
          y: bad.map(function (q) { return curveIvAt(exp, q.k) * 100; }),
          marker: { color: color, symbol: SYMBOLS[i % SYMBOLS.length] + "-open", size: 9, line: { width: 1.5 } },
          hovertemplate: bad.map(quoteHover),
        });
      }
    });
    var sel = state.selected;
    if (sel) {
      var exp = p.expiries.find(function (e) { return e.days === sel.expiry; });
      var pts = exp ? smileQuotes(p, exp).filter(function (q) { return sel.strikes.indexOf(q.strike) >= 0; }) : [];
      traces.push({
        type: "scatter", mode: "markers", name: "selected violation", showlegend: false,
        x: pts.map(function (q) { return q.k; }),
        y: pts.map(function (q) { return (q.iv !== null ? q.iv : curveIvAt(exp, q.k)) * 100; }),
        marker: { size: 18, color: "rgba(0,0,0,0)", line: { color: css("--text"), width: 2 } },
        hoverinfo: "skip",
      });
    }
    var layout = baseLayout("Log-forward moneyness k = ln(K/F)", "Implied volatility (%)");
    Plotly.react(el("smile"), traces, layout, CONFIG);

    var failed = p.quotes.filter(function (q) { return q.smile && q.iv === null; });
    var list = el("failed-list");
    list.innerHTML = "";
    failed.forEach(function (q) {
      var li = document.createElement("li");
      li.textContent = q.expiry.toFixed(0) + "d " + q.type + " K=" + q.strike.toFixed(2) + ": " + q.reason;
      list.appendChild(li);
    });
    el("failed-title").textContent = failed.length
      ? "Failed solves (hollow markers, drawn on the fitted curve):"
      : "Every out-of-the-money quote solved.";

    var near = nearest30(p);
    el("smile-takeaway").textContent = near && near.rmse_vol_pts !== null
      ? near.days.toFixed(0) + "-day SVI fit RMSE: " + near.rmse_vol_pts.toFixed(2) +
        " vol points over " + smileQuotes(p, near).length + " quotes."
      : "Not enough solved quotes to fit a smile.";
  }

  function nearest30(p) {
    var best = null;
    p.expiries.forEach(function (e) {
      if (e.svi && (!best || Math.abs(e.days - 30) < Math.abs(best.days - 30))) best = e;
    });
    return best;
  }

  function renderSurface(p) {
    var fitted = p.expiries.filter(function (e) { return e.svi; });
    if (!fitted.length) return;
    var kmin = Math.max.apply(null, fitted.map(function (e) { return e.curve.k[0]; }));
    var kmax = Math.min.apply(null, fitted.map(function (e) { return e.curve.k[e.curve.k.length - 1]; }));
    var ks = [];
    for (var i = 0; i <= 40; i++) ks.push(kmin + ((kmax - kmin) * i) / 40);
    var z = fitted.map(function (e) {
      return ks.map(function (k) { return curveIvAt(e, k) * 100; });
    });
    var pts = p.quotes.filter(function (q) { return q.smile && q.iv !== null; });
    var axis = function (t) {
      return { title: { text: t }, gridcolor: css("--grid"), backgroundcolor: "rgba(0,0,0,0)", color: css("--text") };
    };
    Plotly.react(
      el("surface"),
      [
        {
          type: "surface", x: ks, y: fitted.map(function (e) { return e.days; }), z: z,
          colorscale: "Viridis", opacity: 0.85, colorbar: { title: { text: "IV (%)" } },
          hovertemplate: "k=%{x:.3f}<br>%{y:.0f} days<br>IV %{z:.2f}%<extra>SVI fit</extra>",
        },
        {
          type: "scatter3d", mode: "markers", name: "quotes",
          x: pts.map(function (q) { return q.k; }), y: pts.map(function (q) { return q.expiry; }),
          z: pts.map(function (q) { return q.iv * 100; }),
          marker: { size: 2.5, color: css("--text") }, hovertemplate: pts.map(quoteHover),
        },
      ],
      {
        paper_bgcolor: "rgba(0,0,0,0)", font: { color: css("--text") }, margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          xaxis: axis("Log-moneyness k"), yaxis: axis("Days to expiry"), zaxis: axis("IV (%)"),
          camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
        },
        showlegend: false,
      },
      CONFIG
    );
  }

  function renderTerm(p) {
    var fitted = p.expiries.filter(function (e) { return e.svi; });
    var days = fitted.map(function (e) { return e.days; });
    var common = { type: "scatter", mode: "lines+markers", x: days, marker: { size: 7 } };
    var atm = Object.assign({}, common, {
      y: fitted.map(function (e) { return e.atm_iv * 100; }),
      line: { color: COLORS[1] }, marker: { symbol: "circle", size: 7 },
      hovertemplate: "%{x:.0f} days<br>ATM IV %{y:.2f}%<extra></extra>",
    });
    var skew = Object.assign({}, common, {
      y: fitted.map(function (e) { return e.skew_25d; }),
      line: { color: COLORS[4], dash: "dash" }, marker: { symbol: "diamond", size: 7 },
      hovertemplate: "%{x:.0f} days<br>25-delta skew %{y:.2f} vol pts<extra></extra>",
    });
    var l1 = baseLayout("Days to expiry", "ATM IV (%)");
    var l2 = baseLayout("Days to expiry", "Put minus call IV (vol pts)");
    l1.showlegend = l2.showlegend = false;
    Plotly.react(el("term"), [atm], l1, CONFIG);
    Plotly.react(el("skew"), [skew], l2, CONFIG);
    if (fitted.length >= 2) {
      var a = fitted[0], b = fitted[fitted.length - 1];
      el("term-takeaway").textContent = "ATM IV goes from " + pct(a.atm_iv) + " at " + a.days.toFixed(0) +
        " days to " + pct(b.atm_iv) + " at " + b.days.toFixed(0) + " days.";
    }
    var near = nearest30(p);
    if (near && near.skew_25d !== null) {
      el("skew-takeaway").textContent = near.days.toFixed(0) + "-day 25-delta skew: " +
        near.skew_25d.toFixed(1) + " vol points.";
    }
  }

  function renderArbs(p) {
    var live = p.source && p.source.kind !== "simulated";
    var rows = p.violations.slice().sort(function (a, b) {
      return state.sortDesc ? b.size - a.size : a.size - b.size;
    });
    el("spread-col").hidden = !live;
    var body = el("arb-body");
    body.innerHTML = "";
    rows.forEach(function (v) {
      var tr = document.createElement("tr");
      var cells = [v.type, v.expiry.toFixed(1) + "d", v.strikes.map(function (k) { return k.toFixed(2); }).join(", "),
        v.size.toFixed(4)];
      if (live) cells.push(v.exceeds_spread === null ? "n/a" : v.exceeds_spread ? "yes" : "no");
      cells.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      if (state.selected && state.selected.expiry === v.expiry && state.selected.type === v.type &&
          state.selected.strikes.join() === v.strikes.join()) tr.className = "selected";
      tr.addEventListener("click", function () {
        state.selected = tr.className === "selected" ? null : v;
        renderArbs(state.payload);
        renderSmile(state.payload);
      });
      body.appendChild(tr);
    });
    var summary = "checked: " + p.checks.conditions + " conditions on " + p.checks.quotes + " quotes";
    el("arb-empty").hidden = rows.length > 0;
    el("arb-empty").textContent = "No static-arbitrage violations this tick (" + summary + ")";
    el("arb-table").hidden = rows.length === 0;
    var byType = {};
    rows.forEach(function (v) { byType[v.type] = (byType[v.type] || 0) + 1; });
    el("arb-takeaway").textContent = rows.length
      ? rows.length + " violations (" + Object.keys(byType).map(function (t) { return byType[t] + " " + t; }).join(", ") +
        "); " + summary + ". Largest: " + rows.slice().sort(function (a, b) { return b.size - a.size; })[0].size.toFixed(4) + " $."
      : "";
  }

  function sourceLabel(src) {
    if (!src) return "";
    if (src.kind === "simulated") return "Simulated, seed " + src.seed + ", IV jitter " + src.jitter;
    return "yfinance " + src.symbol + (src.fetched_at ? ", fetched " + src.fetched_at : "") +
      ", rate " + pct(src.rate, 2);
  }

  function renderHeader(p) {
    el("source").textContent = sourceLabel(p.source);
    el("spot").textContent = p.spot.toFixed(2);
    el("tick").textContent = p.tick;
  }

  function render(p) {
    state.payload = p;
    if (state.selected) {
      var still = p.violations.some(function (v) {
        return v.type === state.selected.type && v.expiry === state.selected.expiry &&
          v.strikes.join() === state.selected.strikes.join();
      });
      if (!still) state.selected = null;
    }
    renderHeader(p);
    renderSmile(p);
    if (el("surface") && state.tab === "surface") renderSurface(p);
    renderTerm(p);
    renderArbs(p);
  }

  function rerender() {
    if (state.payload) render(state.payload);
  }

  function setTab(tab) {
    state.tab = tab;
    el("smile-view").hidden = tab !== "smile";
    el("surface-view").hidden = tab !== "surface";
    document.querySelectorAll("[data-tab]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-tab") === tab));
    });
    rerender();
    window.dispatchEvent(new Event("resize"));
  }

  function setupCommon() {
    document.querySelectorAll("[data-tab]").forEach(function (b) {
      b.addEventListener("click", function () { setTab(b.getAttribute("data-tab")); });
    });
    el("sort-size").addEventListener("click", function () {
      state.sortDesc = !state.sortDesc;
      rerender();
    });
    el("theme").addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.dataset.theme
        ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      rerender();
    });
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
    }
  }

  // Live page: WebSocket feed with pause, stale marking and reconnect.
  function connectLive() {
    var paused = false;
    var pending = null;
    var retryMs = 1000;
    var retryTimer = null;
    var ws = null;

    el("pause").addEventListener("click", function () {
      paused = !paused;
      el("pause").setAttribute("aria-pressed", String(paused));
      el("pause").textContent = paused ? "Resume" : "Pause";
      if (!paused && pending) {
        render(pending);
        pending = null;
      }
    });
    el("reconnect").addEventListener("click", function () {
      retryMs = 1000;
      open();
    });

    function open() {
      clearTimeout(retryTimer);
      el("closed-banner").hidden = true;
      var scheme = location.protocol === "https:" ? "wss://" : "ws://";
      ws = new WebSocket(scheme + location.host + "/ws");
      ws.onopen = function () { retryMs = 1000; };
      ws.onmessage = function (event) {
        var msg = JSON.parse(event.data);
        if (msg.kind === "error") {
          el("error-text").textContent = msg.message;
          el("error-banner").hidden = false;
          el("content").classList.add("stale");
          return;
        }
        el("loading").hidden = true;
        el("content").hidden = false;
        el("error-banner").hidden = true;
        el("content").classList.remove("stale");
        if (paused) pending = msg;
        else render(msg);
      };
      ws.onclose = function () {
        el("closed-banner").hidden = false;
        el("retry-in").textContent = (retryMs / 1000).toFixed(0);
        retryTimer = setTimeout(open, retryMs);
        retryMs = Math.min(retryMs * 2, 30000);
      };
    }
    open();
  }

  window.VolDash = {
    renderStatic: function (payload) {
      setupCommon();
      el("loading").hidden = true;
      el("content").hidden = false;
      render(payload);
    },
    startLive: function () {
      setupCommon();
      connectLive();
    },
  };
})();
