/* volatilityscanner report: charts specific to this page. Data comes from #chart-data,
   which report.py writes from the seeded run; nothing here is hand-typed. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function niceStep(span, n) { var raw = span / n, e = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * e; }

  /* Small multiples of SVI residuals (quote IV minus fit, vol points), one panel per expiry.
     Residuals against the two-thirds fit. Filled: its training strikes. Open: held out. Shared x and y scales. */
  function residuals(el, panels) {
    var last = 0, timer;
    var kmin = Infinity, kmax = -Infinity, rmax = 0;
    panels.forEach(function (p) { p.points.forEach(function (q) { kmin = Math.min(kmin, q[0]); kmax = Math.max(kmax, q[0]); rmax = Math.max(rmax, Math.abs(q[1])); }); });
    var step = niceStep(2 * rmax, 4), ylim = Math.ceil(rmax / step) * step, kstep = niceStep(kmax - kmin, 4);
    function draw() {
      var W = Math.max(200, el.clientWidth);
      if (W === last) return; last = W;
      var cols = W >= 600 ? 2 : 1, gap = 24, pw = Math.floor((W - gap * (cols - 1)) / cols), ph = 190;
      var L = 52, R = 10, T = 28, B = 40;
      var X = function (v) { return L + (pw - L - R) * (v - kmin) / (kmax - kmin); };
      var Y = function (v) { return T + (ph - T - B) * (1 - (v + ylim) / (2 * ylim)); };
      var html = '<div class="legend" aria-hidden="true"><span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--series-1)"/></svg>Training strikes</span><span><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="var(--surface-1)" stroke="var(--series-1)" stroke-width="1.5"/></svg>Held-out strikes</span></div>';
      html += '<div class="yl">Quote IV − fit (vol points)</div><div class="sm" style="grid-template-columns:repeat(' + cols + ',1fr)">';
      panels.forEach(function (p) {
        var s = '<svg xmlns="' + NS + '" class="chart" viewBox="0 0 ' + pw + " " + ph + '" width="' + pw + '" height="' + ph + '" aria-hidden="true">';
        s += '<text class="op" x="' + L + '" y="16">' + esc(p.days + "-day") + '</text><text class="val" x="' + (pw - R) + '" y="16" text-anchor="end">train ' + p.train.toFixed(3) + " · held out " + p.held_out.toFixed(3) + "</text>";
        for (var v = -ylim; v <= ylim + 1e-12; v += step) {
          var y = Y(v);
          s += '<line x1="' + L + '" x2="' + (pw - R) + '" y1="' + y + '" y2="' + y + '" stroke="' + (Math.abs(v) < 1e-12 ? "var(--axis)" : "var(--grid)") + '"/>';
          s += '<text x="' + (L - 8) + '" y="' + (y + 4) + '" text-anchor="end">' + (+v.toFixed(3)).toString().replace("-", "−") + "</text>";
        }
        for (var k = Math.ceil(kmin / kstep) * kstep; k <= kmax + 1e-12; k += kstep) {
          s += '<text x="' + X(k) + '" y="' + (ph - B + 18) + '" text-anchor="middle">' + (+k.toFixed(2)).toString().replace("-", "−") + "</text>";
        }
        s += '<text x="' + ((L + pw - R) / 2) + '" y="' + (ph - 4) + '" text-anchor="middle">Log-moneyness ln(K/F)</text>';
        p.points.forEach(function (q) {
          s += q[2]
            ? '<circle cx="' + X(q[0]) + '" cy="' + Y(q[1]) + '" r="4" fill="var(--surface-1)" stroke="var(--series-1)" stroke-width="2"/>'
            : '<circle cx="' + X(q[0]) + '" cy="' + Y(q[1]) + '" r="4" fill="var(--series-1)"/>';
        });
        html += s + "</svg>";
      });
      el.innerHTML = html + "</div>";
    }
    el.classList.add("plot");
    draw();
    if ("ResizeObserver" in window) new ResizeObserver(function () { clearTimeout(timer); timer = setTimeout(draw, 100); }).observe(el);
  }

  function init() {
    var d = RK.data("chart-data");
    if (!d) return;
    RKCharts.hbar("#c-noise", {
      rows: d.noise.map(function (n) { return { label: n.label, values: { v: n.share } }; }),
      series: [{ key: "v", name: "Ticks flagged", color: "var(--series-1)" }],
      unit: "% of ticks", xmax: 100,
      format: function (v) { return Math.round(v) + " %"; },
      title: "Share of ticks with a static-arbitrage violation, by IV noise sd: " + d.noise.map(function (n) { return n.label + " " + Math.round(n.share) + "%"; }).join(", ")
    });
    residuals(document.getElementById("c-fits"), d.residuals);
    /* Tables render without JS; once the charts exist, show the chart view first. */
    document.querySelectorAll(".figure .rk-table").forEach(function (t) { t.hidden = true; });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
