// Chart rendering shared by the live dashboard and the static report.
(function () {
  "use strict";

  // Okabe-Ito colors, plus a dash pattern and marker shape per expiry so color is never the only cue.
  var SERIES = [
    { color: "#E69F00", dash: "solid", symbol: "circle" },
    { color: "#0072B2", dash: "dash", symbol: "square" },
    { color: "#009E73", dash: "dot", symbol: "diamond" },
    { color: "#CC79A7", dash: "dashdot", symbol: "triangle-up" },
    { color: "#D55E00", dash: "longdash", symbol: "x" },
    { color: "#56B4E9", dash: "longdashdot", symbol: "star" },
  ];

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function baseLayout(xTitle, yTitle) {
    var ink = css("--ink"), muted = css("--muted"), grid = css("--grid");
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, family: css("--font"), size: 12 },
      margin: { l: 56, r: 12, t: 8, b: 56 },
      xaxis: { title: { text: xTitle }, gridcolor: grid, zerolinecolor: muted, color: muted },
      yaxis: { title: { text: yTitle }, gridcolor: grid, zerolinecolor: muted, color: muted },
      legend: { orientation: "h", y: -0.3, font: { color: ink } },
      hoverlabel: { bgcolor: css("--panel"), font: { color: ink } },
    };
  }

  var CONFIG = { responsive: true, displaylogo: false };

  function pct(x, digits) {
    return (x * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function label(days) {
    return Math.round(days) + "-day";
  }

  function nearest(slices, days) {
    return slices.reduce(function (best, s) {
      return Math.abs(s.days - days) < Math.abs(best.days - days) ? s : best;
    });
  }

  // Out-of-the-money quotes: puts below the forward, calls at or above it.
  function otmIndices(p, days) {
    var out = [];
    for (var i = 0; i < p.strike.length; i++) {
      if (p.expiry_days[i] !== days) continue;
      var below = p.k[i] < 0;
      if ((below && p.type[i] === "put") || (!below && p.type[i] === "call")) out.push(i);
    }
    return out;
  }

  function fitAt(slice, k) {
    var s = slice.svi, d = k - s.m;
    var w = s.a + s.b * (s.rho * d + Math.sqrt(d * d + s.s * s.s));
    return Math.sqrt(Math.max(w, 0) / (slice.days / 365));
  }

  function failedQuotes(p) {
    var out = [];
    p.slices.forEach(function (s) {
      otmIndices(p, s.days).forEach(function (i) {
        if (p.iv[i] === null) out.push({ i: i, slice: s });
      });
    });
    return out;
  }

  function smileTraces(p, highlight) {
    var traces = [];
    p.slices.forEach(function (s, n) {
      var st = SERIES[n % SERIES.length], name = label(s.days);
      traces.push({
        type: "scatter", mode: "lines", name: name + " SVI fit", legendgroup: name,
        x: s.curve_k, y: s.curve_iv.map(function (v) { return v * 100; }),
        line: { color: st.color, dash: st.dash, width: 2 },
        hovertemplate: name + " fit<br>k %{x:.3f}<br>IV %{y:.2f}%<extra></extra>",
      });
      var idx = otmIndices(p, s.days), ok = idx.filter(function (i) { return p.iv[i] !== null; });
      var bad = idx.filter(function (i) { return p.iv[i] === null; });
      traces.push({
        type: "scatter", mode: "markers", name: name + " quotes", legendgroup: name, showlegend: false,
        x: ok.map(function (i) { return p.k[i]; }), y: ok.map(function (i) { return p.iv[i] * 100; }),
        marker: { color: st.color, symbol: st.symbol, size: 7 },
        customdata: ok.map(function (i) {
          return [p.strike[i], p.type[i], p.price[i], p.iterations[i]];
        }),
        hovertemplate: name + " %{customdata[1]}<br>strike %{customdata[0]:.2f}<br>price %{customdata[2]:.4f}" +
          "<br>IV %{y:.2f}%<br>iterations %{customdata[3]}<extra></extra>",
      });
      if (bad.length) {
        traces.push({
          type: "scatter", mode: "markers", name: name + " no IV", legendgroup: name, showlegend: false,
          x: bad.map(function (i) { return p.k[i]; }),
          y: bad.map(function (i) { return fitAt(s, p.k[i]) * 100; }),
          marker: { color: st.color, symbol: st.symbol + "-open", size: 9, line: { width: 2 } },
          customdata: bad.map(function (i) { return [p.strike[i], p.type[i], p.reason[i]]; }),
          hovertemplate: name + " %{customdata[1]}<br>strike %{customdata[0]:.2f}<br>no IV: %{customdata[2]}" +
            "<br>(drawn on the fit)<extra></extra>",
        });
      }
    });
    if (highlight) {
      traces.push({
        type: "scatter", mode: "markers", name: "selected violation", showlegend: true,
        x: highlight.k, y: highlight.iv,
        marker: { color: "rgba(0,0,0,0)", size: 18, symbol: "circle-open", line: { color: css("--ink"), width: 2 } },
        hoverinfo: "skip",
      });
    }
    return traces;
  }

  function renderSmile(el, p, highlight) {
    var layout = baseLayout("Log-moneyness k = ln(K/F)", "Implied volatility (%)");
    Plotly.react(el, smileTraces(p, highlight), layout, CONFIG);
  }

  function renderSurface(el, p) {
    if (!p.surface) return;
    var layout = baseLayout("", "");
    var ink = css("--ink");
    layout.margin = { l: 0, r: 0, t: 0, b: 0 };
    layout.scene = {
      xaxis: { title: { text: "Log-moneyness" }, color: ink },
      yaxis: { title: { text: "Days to expiry" }, color: ink },
      zaxis: { title: { text: "IV (%)" }, color: ink },
      camera: { eye: { x: 1.6, y: -1.6, z: 0.9 } },
    };
    var pts = p.iv.map(function (v, i) { return v === null ? null : i; }).filter(function (i) {
      return i !== null && p.k[i] >= p.surface.k[0] && p.k[i] <= p.surface.k[p.surface.k.length - 1];
    });
    Plotly.react(el, [
      { type: "surface", x: p.surface.k, y: p.surface.days, z: p.surface.iv, colorscale: "Viridis",
        colorbar: { title: { text: "IV (%)" } }, name: "SVI fit",
        hovertemplate: "k %{x:.3f}<br>%{y} days<br>IV %{z:.2f}%<extra>SVI fit</extra>" },
      { type: "scatter3d", mode: "markers", name: "quotes",
        x: pts.map(function (i) { return p.k[i]; }), y: pts.map(function (i) { return p.expiry_days[i]; }),
        z: pts.map(function (i) { return p.iv[i] * 100; }),
        marker: { size: 2.5, color: ink }, hovertemplate: "k %{x:.3f}<br>%{y} days<br>IV %{z:.2f}%<extra>quote</extra>" },
    ], layout, CONFIG);
  }

  function renderTerm(el, p) {
    var layout = baseLayout("Days to expiry", "ATM implied volatility (%)");
    layout.showlegend = false;
    Plotly.react(el, [{
      type: "scatter", mode: "lines+markers", x: p.slices.map(function (s) { return s.days; }),
      y: p.slices.map(function (s) { return s.atm_iv * 100; }), line: { color: SERIES[1].color },
      marker: { symbol: "square", size: 8 }, hovertemplate: "%{x} days<br>ATM IV %{y:.2f}%<extra></extra>",
    }], layout, CONFIG);
  }

  function renderSkew(el, p) {
    var layout = baseLayout("Days to expiry", "Put minus call IV (vol pts)");
    layout.showlegend = false;
    Plotly.react(el, [{
      type: "scatter", mode: "lines+markers", x: p.slices.map(function (s) { return s.days; }),
      y: p.slices.map(function (s) { return s.skew_25d; }), line: { color: SERIES[0].color, dash: "dash" },
      marker: { symbol: "circle", size: 8 }, hovertemplate: "%{x} days<br>skew %{y:.2f} vol points<extra></extra>",
    }], layout, CONFIG);
  }

  function takeaways(p) {
    if (!p.slices.length) return { smile: "No expiry had enough solved quotes to fit.", term: "", skew: "" };
    var m = nearest(p.slices, 30), first = p.slices[0], last = p.slices[p.slices.length - 1];
    var withSkew = p.slices.filter(function (s) { return s.skew_25d !== null; });
    var widest = withSkew.length ? withSkew.reduce(function (a, b) { return b.skew_25d > a.skew_25d ? b : a; }) : null;
    var mSkew = m.skew_25d === null ? "n/a (25-delta strike outside the quotes)" : m.skew_25d.toFixed(1) + " vol points";
    var failed = failedQuotes(p).length;
    return {
      smile: label(m.days) + " 25-delta skew: " + mSkew + "; SVI fit RMSE " +
        m.rmse_vol_points.toFixed(2) + " vol points" + (failed ? "; " + failed + " quote(s) without an IV." : "."),
      term: "ATM IV goes from " + pct(first.atm_iv) + " at " + first.days + " days to " + pct(last.atm_iv) +
        " at " + last.days + " days.",
      skew: widest ? "25-delta skew is widest at " + widest.days + " days: " + widest.skew_25d.toFixed(1) +
        " vol points." : "No expiry quotes both 25-delta strikes.",
      surface: "Fitted surface over k from " + p.surface.k[0].toFixed(3) + " to " +
        p.surface.k[p.surface.k.length - 1].toFixed(3) + " across " + p.slices.length + " expiries.",
    };
  }

  function renderFailed(el, p) {
    var rows = failedQuotes(p);
    el.innerHTML = "";
    if (!rows.length) {
      el.textContent = "Every out-of-the-money quote solved.";
      return;
    }
    rows.forEach(function (r) {
      var li = document.createElement("li");
      li.textContent = label(r.slice.days) + " " + p.type[r.i] + " " + p.strike[r.i].toFixed(2) +
        ": no IV (" + p.reason[r.i] + ")";
      el.appendChild(li);
    });
  }

  function money(x) {
    return "$" + x.toFixed(x < 0.01 ? 6 : 4);
  }

  // onSelect(violation) is called when a row is clicked; sortDesc toggles with the Size header.
  function renderArbTable(container, p, state, onSelect) {
    var v = p.violations.slice();
    container.innerHTML = "";
    if (!v.length) {
      var empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No static-arbitrage violations this tick (checked: " + p.conditions_checked +
        " conditions on " + p.quotes_checked + " quotes)";
      container.appendChild(empty);
      return;
    }
    var live = v.some(function (x) { return x.exceeds_spread !== null; });
    v.sort(function (a, b) { return state.sortDesc ? b.size - a.size : a.size - b.size; });
    var table = document.createElement("table");
    var head = "<thead><tr><th scope='col'>Type</th><th scope='col'>Expiry</th><th scope='col'>Strikes</th>" +
      "<th scope='col'><button type='button' class='sort'>Size ($) " + (state.sortDesc ? "&darr;" : "&uarr;") +
      "</button></th>" + (live ? "<th scope='col'>Exceeds spread</th>" : "") + "</tr></thead>";
    table.innerHTML = head + "<tbody></tbody>";
    var body = table.querySelector("tbody");
    v.forEach(function (x) {
      var tr = document.createElement("tr");
      tr.tabIndex = 0;
      var cells = [x.type.replace(/_/g, " "), label(x.expiry_days),
        x.strikes.map(function (k) { return k.toFixed(2); }).join(" / "), money(x.size)];
      if (live) cells.push(x.exceeds_spread === null ? "n/a" : x.exceeds_spread ? "yes" : "no");
      cells.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      function select() { onSelect(x); }
      tr.addEventListener("click", select);
      tr.addEventListener("keydown", function (e) { if (e.key === "Enter") select(); });
      body.appendChild(tr);
    });
    table.querySelector(".sort").addEventListener("click", function () {
      state.sortDesc = !state.sortDesc;
      renderArbTable(container, p, state, onSelect);
    });
    container.appendChild(table);
  }

  // Points to circle on the smile for a violation: its strikes at its expiry, on the fit.
  function highlightFor(p, violation) {
    var s = p.slices.filter(function (x) { return x.days === violation.expiry_days; })[0];
    if (!s) return null;
    var ks = violation.strikes.map(function (K) { return Math.log(K / s.forward); });
    return { k: ks, iv: ks.map(function (k) { return fitAt(s, k) * 100; }) };
  }

  window.VS = {
    renderSmile: renderSmile, renderSurface: renderSurface, renderTerm: renderTerm, renderSkew: renderSkew,
    renderFailed: renderFailed, renderArbTable: renderArbTable, highlightFor: highlightFor, takeaways: takeaways,
  };
})();
