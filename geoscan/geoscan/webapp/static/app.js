"use strict";
const $ = (s) => document.querySelector(s);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return r;
};
const getJSON = async (url) => (await api(url)).json();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = { config: null, blocks: [], block: null, options: {}, params: {}, element: "cu", run: null, layer: "cpi", jobTimer: null, fitted: null };
const CONTOUR_COLORS = ["#ffffff", "#ffeb3b", "#ff1744"];
const TIER_COLORS = { "1": "#d50000", "2": "#ff9100", "3": "#9e9e9e" };

/* ---------------- map ---------------- */
const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([21, 79], 6);
const bases = {
  sat: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 19, attribution: "Imagery © Esri, Maxar, Earthstar Geographics" }),
  topo: L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    { maxZoom: 17, attribution: "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)" }),
};
bases.sat.addTo(map);
L.control.scale({ imperial: false }).addTo(map);
const G = { overlay: null, na: null, contours: null, stations: null, targets: null, lineaments: null, zones: null, aoi: null, targetMarkers: {} };

function clearLayer(k) { if (G[k]) { map.removeLayer(G[k]); G[k] = null; } }

async function drawMap() {
  const run = S.run;
  ["overlay", "na", "contours", "stations", "targets", "lineaments", "zones", "aoi"].forEach(clearLayer);
  if (!run) { renderLegend(null); return; }
  const id = run.run_id;
  const aoi = await getJSON(`/api/runs/${id}/vector/aoi.geojson`);
  G.aoi = L.geoJSON(aoi, { style: { color: "#00e5ff", weight: 2, fill: false }, interactive: false }).addTo(map);
  if (S.fitted !== run.block) { map.fitBounds(G.aoi.getBounds(), { padding: [30, 30] }); S.fitted = run.block; }
  await drawRaster();
  await drawVectors();
}

const refOf = (layer) => layer.startsWith("ref:") ? S.run.reference.find((r) => r.key === layer.slice(4)) : null;

function rampColour(colors, t) {
  return colors[Math.max(0, Math.min(colors.length - 1, Math.round(t * (colors.length - 1))))];
}

const GEOCHEM_RAMP = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"];

function showNotice(html) {
  const el = $("#map-notice");
  el.innerHTML = html || "";
  el.hidden = !html;
}

/* a reference layer with no data for this area: say so, and why */
function drawUnavailable(ref) {
  showNotice(`<b>${esc(ref.label)}</b><br>Not available for this area.<br><span class="small">${esc(ref.reason || "")}</span>`);
  return { kind: "note", label: ref.label, text: "Not available for this area." };
}

/* reference geochemistry: stream sediment samples as squares, coloured and sized by the chosen element; no surface */
async function drawGeochem(ref) {
  const el = ref.elements.find((e) => e.key === S.element) || ref.elements[0];
  S.element = el.key;
  const gj = await getJSON(`/api/blocks/${S.run.block}/geochem/samples.geojson`);
  const span = (el.max - el.min) || 1;
  const quick = ["cu", "zn", "pb", "au", "as", "ni"].filter((k) => ref.elements.some((e) => e.key === k));
  G.stations = L.geoJSON(gj, {
    pointToLayer: (f, ll) => {
      const t = (f.properties[el.key] - el.min) / span, px = Math.round(10 + 14 * t);
      return L.marker(ll, { icon: L.divIcon({ className: "", iconSize: [px, px], iconAnchor: [px / 2, px / 2],
        html: `<span class="smp" style="width:${px}px;height:${px}px;background:${rampColour(GEOCHEM_RAMP, t)}"></span>` }) });
    },
    onEachFeature: (f, l) => l.bindTooltip(`<b>${esc(el.label)}: ${f.properties[el.key]}</b><br>sample ${esc(f.properties.sample)}`)
      .bindPopup(`<b>Sample ${esc(f.properties.sample)}</b><table>${[el.key, ...quick.filter((k) => k !== el.key)].map((k) =>
        `<tr><td>${esc(ref.elements.find((e) => e.key === k).label)}</td><td><b>${f.properties[k]}</b></td></tr>`).join("")}</table>
        <div class="small muted">Reflects the ground upstream of this point. The download has all ${ref.elements.length} elements.</div>`),
  }).addTo(map);
  return { kind: "ramp", label: `${el.label} in stream sediment`, unit: "", min: el.min, max: el.max, colors: GEOCHEM_RAMP, levels: [],
    geochem: true, samples: ref.samples, note: ref.note };
}

/* reference geophysics: surface (if any), black contours, stations coloured by value; never part of the CPI */
async function drawReference(ref) {
  const base = `/api/blocks/${S.run.block}/reference/${ref.key}`;
  const info = await getJSON(`${base}/overlay`);
  const lg = info.legend;
  if (!info.none) {
    G.overlay = L.imageOverlay(info.url, info.bounds, { opacity: +$("#opacity").value, interactive: false }).addTo(map);
    if ($("#show-contours").checked) {
      G.contours = L.geoJSON(await getJSON(`${base}/contours.geojson`), {
        style: { color: "#111", weight: 1.2, opacity: 0.85 },
        onEachFeature: (f, l) => l.bindTooltip(`${f.properties.level} ${f.properties.unit}`, { sticky: true }),
      }).addTo(map);
    }
  }
  G.stations = L.geoJSON(await getJSON(`${base}/stations.geojson`), {
    // diamonds, so stations are not mistaken for field-check locations (circles)
    pointToLayer: (f, ll) => L.marker(ll, { icon: L.divIcon({ className: "", iconSize: [14, 14], iconAnchor: [7, 7],
      html: `<span class="stn" style="background:${rampColour(lg.colors, (f.properties.value - lg.min) / (lg.max - lg.min || 1))}"></span>` }) }),
    onEachFeature: (f, l) => l.bindTooltip(`<b>${f.properties.value} ${esc(f.properties.unit)}</b><br>station ${f.properties.station}` +
      (f.properties.elevation_m != null ? `<br>elevation ${f.properties.elevation_m} m` : "")),
  }).addTo(map);
  return { ...lg, stations: ref.stations, spacing_m: ref.spacing_m, has_surface: ref.has_surface };
}

async function drawRaster() {
  clearLayer("overlay"); clearLayer("contours"); clearLayer("na"); clearLayer("stations");
  const run = S.run; if (!run) return;
  let legend = null;
  const ref = refOf(S.layer);
  showNotice("");
  if (ref) {
    try {
      legend = !ref.available ? drawUnavailable(ref) : ref.kind === "samples" ? await drawGeochem(ref) : await drawReference(ref);
    } catch (e) {            // never leave the map silently empty
      showNotice(`<b>${esc(ref.label)}</b><br>This layer could not be loaded.<br><span class="small">${esc(e.message)}. Reload the page and try again.</span>`);
    }
    S.hatchShown = false;
    S.legend = legend;
    renderLegend(legend);
    G.aoi && G.aoi.bringToFront();
    G.targets && G.targets.bringToFront();
    return;
  }
  if (S.layer !== "none") {
    const info = await getJSON(`/api/runs/${run.run_id}/overlay/${S.layer}`);
    G.overlay = L.imageOverlay(info.url, info.bounds, { opacity: +$("#opacity").value, interactive: false }).addTo(map);
    legend = info.legend;
    if ($("#show-contours").checked && info.legend.levels && info.legend.levels.length) {
      const gj = await getJSON(`/api/runs/${run.run_id}/contours/${S.layer}.geojson`);
      const lv = info.legend.levels;
      G.contours = L.geoJSON(gj, {
        style: (f) => ({ color: CONTOUR_COLORS[lv.indexOf(f.properties.level)] || "#fff", weight: 2, opacity: 0.95 }),
        onEachFeature: (f, l) => l.bindTooltip(`${f.properties.level} ${f.properties.unit}`, { sticky: true }),
      }).addTo(map);
    }
  }
  S.hatchShown = false;
  if ($("#show-na").checked && S.layer !== "none") {
    // per layer: only EMIT-based layers can be "not assessed", and only where nothing was detected
    const na = await getJSON(`/api/runs/${run.run_id}/overlay/na_${S.layer}`);
    if (!na.none) {
      G.na = L.imageOverlay(na.url, na.bounds, { opacity: 1, interactive: false }).addTo(map);
      S.hatchShown = true;
    }
  }
  S.legend = legend;
  renderLegend(legend);
  G.aoi && G.aoi.bringToFront();
}

async function drawVectors() {
  ["targets", "lineaments", "zones"].forEach(clearLayer);
  const id = S.run.run_id;
  G.zoneLayers = {};
  if ($("#show-zones").checked) {
    try {
      const gj = await getJSON(`/api/runs/${id}/vector/zones.geojson`);
      G.zones = L.geoJSON(gj, {
        style: (f) => {
          const t = String(f.properties.Priority_Tier || "").match(/(\d)/)?.[1];
          return { color: TIER_COLORS[t] || "#999", weight: t === "3" ? 1.5 : 2.5, dashArray: t === "3" ? "5 4" : null,
            fillColor: TIER_COLORS[t] || "#999", fillOpacity: 0.08 };
        },
        onEachFeature: (f, l) => {
          G.zoneLayers[f.properties.Zone_ID] = l;
          l.bindPopup(zonePopup(f.properties));
          if (f.properties.Zone <= 12) l.bindTooltip(f.properties.Zone_ID, { permanent: true, direction: "center", className: "zone-label" });
        },
      }).addTo(map);
    } catch (e) { /* results made before zones existed have no zone file */ }
  }
  if ($("#show-lineaments").checked) {
    const gj = await getJSON(`/api/runs/${id}/vector/lineaments.geojson`);
    G.lineaments = L.geoJSON(gj, {
      style: (f) => f.properties.morphology === "scarp" ? { color: "#bdbdbd", weight: 1.5, dashArray: "5 4" }
        : { color: "#00e676", weight: 2 },
      onEachFeature: (f, l) => l.bindTooltip(`${f.properties.line_id || ""} · strike ${f.properties.strike_deg}° · ` +
        `${Math.round(f.properties.length_m)} m · ${f.properties.morphology}` +
        (f.properties.morphology === "scarp" ? " (not used for CPI)" : "")),
    }).addTo(map);
  }
  if ($("#show-targets").checked) {
    const gj = await getJSON(`/api/runs/${id}/vector/targets.geojson`);
    G.targetMarkers = {};
    G.targets = L.geoJSON(gj, {
      pointToLayer: (f, ll) => {
        const t = String(f.properties.Priority_Tier || "").match(/(\d)/)?.[1];
        const m = L.circleMarker(ll, { radius: f.properties.Rank <= 10 ? 8 : 6, color: "#000", weight: 1,
          fillColor: TIER_COLORS[t] || "#999", fillOpacity: 0.95 });
        G.targetMarkers[f.properties.Rank] = m;
        return m;
      },
      onEachFeature: (f, l) => l.bindPopup(targetPopup(f.properties)),
    }).addTo(map);
  }
  G.aoi && G.aoi.bringToFront();
}

function zonePopup(p) {
  const rows = [["Tier", p.Priority_Tier], ["Area", `${p.Area_km2} km²`], ["Peak / mean CPI", `${p.CPI_Max_Pct} % / ${p.CPI_Mean_Pct} %`],
    ["Main evidence", p.Main_Evidence], ["Muscovite · Quartz", `${p.Muscovite_Pct} % · ${p.Quartz_Pct} %`],
    ["Gossan · Chlorite", `${p.Gossan_Pct} % · ${p.Chlorite_Pct} %`], ["Lineaments inside", `${p.Lineaments_km} km`],
    ["Field-check locations", p.Field_Check_Locations], ["Based on", p.Evidence_Basis]];
  return `<b>Zone ${esc(p.Zone_ID)}</b><table>${rows.map(([k, v]) => `<tr><td>${k}</td><td><b>${esc(v)}</b></td></tr>`).join("")}</table>
    <div class="small muted">Suggested follow-up: ${esc(p.Suggested_Follow_Up)}</div>`;
}

function targetPopup(p) {
  const rows = [["Zone", p.Zone], ["CPI", `${(+p.CPI_Percent).toFixed(1)} %`], ["Tier", p.Priority_Tier], ["Muscovite", `${p.Muscovite_Pct} %`],
    ["Quartz", `${p.Quartz_Pct} %`], ["Gossan", `${p.Gossan_Pct} %`], ["Chlorite", `${p.Chlorite_Pct} %`],
    ["To lineament", `${p.Distance_to_Lineament_m} m`], ["Lat / Lon", `${(+p.Latitude).toFixed(4)}, ${(+p.Longitude).toFixed(4)}`]]
    .filter(([, v]) => v != null && v !== "undefined %" && v !== "");
  const follow = p.Suggested_Follow_Up || "";
  return `<b>#${p.Rank} ${esc(p.Location_Name || p.Target_Name || p.Location_ID || p.Target_ID)}</b>
    <div class="small muted">Suggested field-check location${p.Check_Within_m ? `: look within about ${p.Check_Within_m} m` : ""}. Not a drill collar.</div>
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td><b>${esc(v)}</b></td></tr>`).join("")}</table>` +
    (follow ? `<div class="small muted">Suggested follow-up: ${esc(follow)}</div>` : "");
}

function renderLegend(lg) {
  const el = $("#legend");
  let h = "";
  if (lg && lg.kind === "note") h += `<b>${esc(lg.label)}</b><div class="small" style="color:#b5532a">${esc(lg.text)}</div>`;
  if (lg && lg.kind === "ramp") {
    h += `<b>${esc(lg.label)}</b><div class="ramp" style="background:linear-gradient(90deg,${lg.colors.join(",")})"></div>
      <div class="ticks"><span>${lg.min}</span><span>${lg.unit}</span><span>${lg.max}</span></div>`;
    if (lg.geochem) {
      h += `<div class="row"><span class="smp" style="width:11px;height:11px;background:#fd8d3c;margin:0 2px"></span>${lg.samples} samples · bigger and darker = higher</div>
        <div class="small muted">Points only: each sample reflects the ground upstream of it. Units not stated in the source.</div>` +
        (lg.note ? `<div class="small muted">Highs can reflect rock type (see the i button).</div>` : "") +
        `<div class="small" style="color:#b5532a"><b>Reference only</b>: not used in the CPI.</div>`;
    }
    if (lg.reference) {
      h += `<div class="row"><span class="stn" style="background:#fff;margin:0 3px"></span>
        ${lg.stations} stations${lg.spacing_m ? `, about ${(lg.spacing_m / 1000).toFixed(1)} km apart` : ""}</div>` +
        (lg.has_surface ? ($("#show-contours").checked && lg.contour_step ? `<div class="row"><span class="sw" style="background:#111"></span>contours every ${lg.contour_step} ${lg.unit}</div>` : "") +
          `<div class="small muted">Surface interpolated between stations.</div>`
          : `<div class="small muted">Too few stations for a surface: points only.</div>`) +
        `<div class="small" style="color:#b5532a"><b>Reference only</b>: not used in the CPI.</div>`;
    }
    if ($("#show-contours").checked && lg.levels.length)
      h += lg.levels.map((v, i) => `<div class="row"><span class="sw" style="background:${CONTOUR_COLORS[i]};outline:1px solid #888"></span>${v} ${lg.unit} contour</div>`).join("");
  }
  if ($("#show-targets").checked)
    h += `<div class="row"><span class="sw" style="background:${TIER_COLORS[1]};height:10px;width:10px;border-radius:50%"></span>Tier 1
      <span class="sw" style="background:${TIER_COLORS[2]};height:10px;width:10px;border-radius:50%"></span>Tier 2
      <span class="sw" style="background:${TIER_COLORS[3]};height:10px;width:10px;border-radius:50%"></span>Tier 3 field-check locations</div>`;
  if ($("#show-zones").checked && G.zones)
    h += `<div class="row"><span class="sw" style="height:10px;width:16px;background:rgba(213,0,0,.1);outline:2px solid ${TIER_COLORS[1]}"></span>Prospective zone, coloured by tier</div>`;
  if ($("#show-lineaments").checked)
    h += `<div class="row"><span class="sw" style="background:#00e676"></span>Lineament
      <span class="sw" style="background:repeating-linear-gradient(90deg,#bdbdbd 0 5px,transparent 5px 9px)"></span>Scarp</div>`;
  if ($("#show-na").checked && S.hatchShown) h += `<div class="row"><span class="hatch"></span>Not assessed (no usable EMIT data)</div>`;
  el.innerHTML = h;
  el.hidden = !h;
}

/* ---------------- info pop-ups ---------------- */
const TAG_TEXT = { EXP: "Expert choice (no external source)", ASM: "Project assumption", LIB: "Spectral library",
  LIT: "Published literature", TEST: "Tested against ground truth" };

function showInfo(key) {
  const p = S.config.params[key], o = S.config.options[key];
  const d = (p || o || {}).info || {};
  const meta = [];
  if (p) {
    const dflt = p.kind === "bool" ? (p.default ? "yes" : "no") : `${p.default}${p.unit ? " " + p.unit : ""}`;
    meta.push(`Default: <b>${esc(dflt)}</b>`);
    if (p.kind !== "bool") meta.push(`Allowed: ${p.min} – ${p.max}${p.unit ? " " + esc(p.unit) : ""}`);
    meta.push(`Source: <b>${p.tag}</b> · ${esc(TAG_TEXT[p.tag] || "")}`);
    meta.push(`Group: ${esc(p.group)}`);
  } else if (o) {
    const c = o.choices.find((x) => String(x.value) === String(o.default));
    meta.push(`Recommended: <b>${esc(c ? c.label : o.default)}</b>`);
  }
  const sec = (title, text, cls = "") => text ? `<div class="info-sec ${cls}"><h4>${title}</h4><p>${esc(text)}</p></div>` : "";
  let body = `<div class="info-meta">${meta.map((m) => `<span>${m}</span>`).join("")}</div>`;
  body += sec("What it is", d.what);
  body += sec("How the method uses it", d.how, "formula");
  if (d.choices) body += sec("The choices", d.choices);
  if (d.up || d.down) {
    const upLabel = p && p.kind === "bool" ? "When on" : o ? "Effect" : "If you raise it";
    const downLabel = p && p.kind === "bool" ? "When off" : o ? "Otherwise" : "If you lower it";
    body += `<div class="info-sec"><div class="updown"><div><b>${upLabel}</b>${esc(d.up || "")}</div>
      <div><b>${downLabel}</b>${esc(d.down || "")}</div></div></div>`;
  }
  body += sec("Why this default", d.why);
  body += sec("What our tests show", d.evidence, "evidence");
  if (p && p.help && !d.how) body += sec("Note", p.help);
  $("#info-title").textContent = (p || o).label;
  $("#info-body").innerHTML = body;
  $("#info").showModal();
}

function showLayerInfo(key) {
  const ref = S.run && key.startsWith("ref:") ? refOf(key) : null;
  if (ref && (!ref.available || ref.kind === "samples")) {
    $("#info-title").textContent = ref.label;
    $("#info-body").innerHTML = (ref.available
      ? `<div class="info-meta"><span><b>${ref.samples}</b> samples</span>${ref.spacing_m ? `<span>about <b>${(ref.spacing_m / 1000).toFixed(1)} km</b> apart</span>` : ""}
          <span><b>${ref.elements.length}</b> elements</span></div>`
      : `<div class="info-sec evidence"><h4>Not available for this area</h4><p>${esc(ref.reason || "")}</p></div>`) +
      `<div class="info-sec"><h4>What it is</h4><p>${esc(ref.what)}</p></div>
      <div class="info-sec"><h4>How to read it</h4><p>${esc(ref.caveat)}</p></div>` +
      (ref.note ? `<div class="info-sec evidence"><h4>In this area</h4><p>${esc(ref.note)}</p></div>` : "") +
      `<div class="info-sec"><h4>Source</h4><p>${esc(ref.source)}</p></div>`;
    $("#info").showModal();
    return;
  }
  if (ref) {
    $("#info-title").textContent = ref.label;
    $("#info-body").innerHTML = `<div class="info-meta"><span><b>${ref.stations}</b> stations</span>
      ${ref.spacing_m ? `<span>about <b>${(ref.spacing_m / 1000).toFixed(1)} km</b> apart</span>` : ""}
      <span>range <b>${ref.min} to ${ref.max} ${esc(ref.unit)}</b></span></div>
      <div class="info-sec"><h4>What it is</h4><p>${esc(ref.what)}</p></div>
      <div class="info-sec"><h4>What you see on the map</h4><p>${ref.has_surface
        ? "The dots are the measured stations (hover for the value). The coloured surface and the contour lines are interpolated between them."
        : "Only the measured stations are shown: there are too few to draw a surface."}</p></div>
      <div class="info-sec evidence"><h4>How to read it</h4><p>${esc(ref.caveat)} A zone is a few hundred metres across, so a pattern drawn from stations ${ref.spacing_m ? "about " + (ref.spacing_m / 1000).toFixed(1) + " km" : "this far"} apart cannot confirm or reject a single zone.</p></div>
      <div class="info-sec"><h4>Source</h4><p>${esc(ref.source)}</p></div>`;
    $("#info").showModal();
    return;
  }
  $("#info-title").textContent = S.config.layers[key];
  $("#info-body").innerHTML = `<div class="info-sec"><p>${esc(S.config.layer_info[key] || "")}</p></div>`;
  $("#info").showModal();
}

function infoButton(key, kind = "param") {
  return `<button type="button" class="ibtn" data-info="${esc(key)}" data-kind="${kind}" title="What is this?" aria-label="Explain">i</button>`;
}

document.addEventListener("click", (ev) => {
  const b = ev.target.closest(".ibtn[data-info]");
  if (!b) return;
  ev.preventDefault();
  ev.stopPropagation();
  if (b.dataset.kind === "layer") showLayerInfo(b.dataset.info);
  else showInfo(b.dataset.info);
}, true);
$("#info-close").onclick = () => $("#info").close();
$("#info").addEventListener("click", (ev) => { if (ev.target.id === "info") $("#info").close(); });

/* ---------------- sidebar ---------------- */
function renderOptions() {
  const box = $("#options");
  box.innerHTML = "";
  for (const [key, o] of Object.entries(S.config.options)) {
    const id = "opt-" + key.replace(/\./g, "-");
    const d = document.createElement("div");
    d.className = "opt";
    d.innerHTML = `<label for="${id}">${esc(o.label)}${infoButton(key, "option")}</label><select id="${id}">${o.choices.map((c) =>
      `<option value="${esc(c.value)}">${esc(c.label)}${c.value === o.default ? " ★" : ""}</option>`).join("")}</select>`;
    box.appendChild(d);
    const sel = d.querySelector("select");
    sel.value = String(S.options[key]);
    sel.onchange = () => {
      const c = o.choices.find((x) => String(x.value) === sel.value);
      S.options[key] = c.value;
      selectRun();
    };
  }
}

function sameOptions(a, b) { return Object.keys(S.config.options).every((k) => String(a[k]) === String(b[k])); }
function sameParams(a, b) { return Object.keys(S.config.params).every((k) => Number(a[k]) === Number(b[k])); }
function sameRun(r) { return sameOptions(r.options, S.options) && sameParams(r.params, S.params); }
function defaults(obj) { return Object.fromEntries(Object.entries(obj).map(([k, o]) => [k, o.default])); }

const STEPS = ["acquire", "emit_l2a", "emit_l2b", "aster", "structure", "alteration", "targets", "maps"];
const RERUN_TEXT = {
  targets: "Only the CPI, zones, field-check locations and maps are recomputed (seconds).",
  alteration: "Alteration layers, CPI, zones and maps are recomputed (about a minute).",
  structure: "Lineaments, alteration, CPI, zones and maps are recomputed (a few minutes).",
  emit_l2a: "EMIT detection is redone from the downloaded scenes (several minutes to an hour, depending on area size).",
  acquire: "Full analysis including data download.",
};

function firstStep() {
  const steps = [];
  for (const [k, o] of Object.entries(S.config.options)) if (String(S.options[k]) !== String(o.default)) steps.push(o.step);
  for (const [k, p] of Object.entries(S.config.params)) if (Number(S.params[k]) !== Number(p.default)) steps.push(p.step);
  return steps.sort((a, b) => STEPS.indexOf(a) - STEPS.indexOf(b))[0] || null;
}

function weightSum() {
  return ["muscovite", "quartz", "gossan", "chlorite"].reduce((a, m) => a + Number(S.params["targets.w_" + m]), 0);
}

function ruleProblems() {
  const v = S.params, out = [];
  const w = weightSum();
  if (Math.abs(w - 1) > 0.005) out.push(`Mineral weights add up to ${w.toFixed(2)} (must be 1.00).`);
  if (!(v["targets.tier1"] > v["targets.tier2"] && v["targets.tier2"] > v["targets.tier3"])) out.push("Tiers must decrease: Tier 1 > Tier 2 > Tier 3.");
  if (!(v["alteration.sigma_halo_m"] > v["alteration.sigma_core_m"])) out.push("Halo σ must be larger than core σ.");
  if (!(v["alteration.sericite_max_nm"] > v["alteration.sericite_min_nm"])) out.push("White-mica window: 'to' must be above 'from'.");
  if (!(v["structure.canny_high_quantile"] > v["structure.canny_low_quantile"])) out.push("High edge quantile must be above the low one.");
  if (!(v["structure.min_length_m"] >= v["structure.min_segment_m"])) out.push("Minimum lineament length must be at least the minimum Hough segment.");
  for (const [k, p] of Object.entries(S.config.params)) {
    if (p.kind !== "bool" && !(Number(v[k]) >= p.min && Number(v[k]) <= p.max)) out.push(`${p.label} must be between ${p.min} and ${p.max}.`);
  }
  return out;
}

function renderParams() {
  const box = $("#params");
  const groups = {};
  for (const [k, p] of Object.entries(S.config.params)) (groups[p.group] = groups[p.group] || []).push([k, p]);
  const first = !box.querySelector("details");
  const open = new Set([...box.querySelectorAll("details[open]")].map((d) => d.dataset.group));
  if (first) open.add(Object.keys(groups)[0]);
  box.innerHTML = Object.entries(groups).map(([g, ps]) => {
    const rows = ps.map(([k, p]) => {
      const val = S.params[k], ch = Number(val) !== Number(p.default);
      const input = p.kind === "bool" ? `<input type="checkbox" data-k="${k}"${val ? " checked" : ""}>`
        : `<input type="number" data-k="${k}" value="${val}" min="${p.min}" max="${p.max}" step="${p.step_size}">`;
      const dflt = p.kind === "bool" ? (p.default ? "yes" : "no") : p.default;
      return `<div class="prow${ch ? " changed" : ""}" title="${esc(p.help || "")}">
        <div class="pl">${esc(p.label)}<span class="tag">${p.tag}</span>${infoButton(k)}<span class="def">default ${dflt}${p.unit ? " " + esc(p.unit) : ""}</span></div>
        ${input}${ch ? `<button type="button" class="reset" data-reset="${k}">reset</button>` : ""}</div>`;
    }).join("");
    const sum = g.startsWith("CPI") ? '<div id="wsum" class="psum"></div>' : "";
    return `<details class="pgroup" data-group="${esc(g)}"${open.has(g) ? " open" : ""}><summary>${esc(g)}</summary>${sum}${rows}</details>`;
  }).join("");
  box.querySelectorAll("input[data-k]").forEach((i) => i.onchange = () => {
    const p = S.config.params[i.dataset.k];
    S.params[i.dataset.k] = p.kind === "bool" ? i.checked : Number(i.value);
    renderParams();
    selectRun();
  });
  box.querySelectorAll("[data-reset]").forEach((b) => b.onclick = () => {
    S.params[b.dataset.reset] = S.config.params[b.dataset.reset].default;
    renderParams();
    selectRun();
  });
  const w = weightSum(), ws = $("#wsum");
  if (ws) { ws.className = "psum " + (Math.abs(w - 1) <= 0.005 ? "ok" : "bad"); ws.textContent = `Mineral weights sum: ${w.toFixed(2)} (must be 1.00)`; }
  const n = Object.entries(S.config.params).filter(([k, p]) => Number(S.params[k]) !== Number(p.default)).length;
  $("#params-changed").hidden = !n;
  $("#params-changed").textContent = `${n} changed`;
}

async function refreshBlocks(selectName) {
  S.blocks = await getJSON("/api/blocks");
  const sel = $("#block-select");
  sel.innerHTML = S.blocks.map((b) => `<option value="${esc(b.block)}">${esc(b.block)}${b.area_km2 ? ` · ${b.area_km2} km²` : ""}</option>`).join("");
  const name = selectName || (S.block && S.block.block) || (S.blocks[0] && S.blocks[0].block);
  if (name) sel.value = name;
  S.block = S.blocks.find((b) => b.block === sel.value) || null;
}

function renderBlockInfo() {
  const b = S.block;
  $("#block-info").textContent = b ? `${b.runs.length} result set(s)` + (b.has_ground_truth ? " · ground truth available" : "") +
    (b.uploaded_by ? ` · uploaded by ${b.uploaded_by}` : "") : "";
  const box = $("#computed");
  if (!b || !b.runs.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="title">Already computed for this area</div>` + b.runs.map((r) =>
    `<button class="chip${S.run && S.run.run_id === r.run_id ? " active" : ""}" data-run="${esc(r.run_id)}" title="${esc(r.label)}">
      ${r.recommended ? '<span class="rec">Recommended · </span>' : ""}${esc(Object.entries(r.options).map(([k, v]) =>
        (S.config.options[k].choices.find((c) => String(c.value) === String(v)) || { label: v }).label).join(" · "))}
      ${r.custom ? `<br><span class="small muted">Custom: ${esc(r.label.split("Custom parameters: ")[1] || "")}</span>` : ""}
      ${r.created_by ? `<br><span class="small muted">by ${esc(r.created_by)}</span>` : ""}</button>`).join("");
  box.querySelectorAll(".chip").forEach((c) => c.onclick = () => {
    const r = b.runs.find((x) => x.run_id === c.dataset.run);
    Object.assign(S.options, r.options);
    Object.assign(S.params, r.params);
    renderOptions();
    renderParams();
    selectRun();
  });
}

async function selectRun() {
  const b = S.block;
  const st = $("#run-status"), btn = $("#run-btn"), bar = $("#run-bar");
  const match = b && b.runs.find(sameRun);
  const job = await activeJob();
  if (match) {
    st.className = "status ready";
    st.textContent = match.recommended ? "Results ready (recommended method)." : match.custom ? "Results ready (custom parameters)." : "Results ready.";
    bar.hidden = true;
    // remember what is being viewed, so "Discard changes" can come back to it
    S.last = { block: b.block, options: { ...S.options }, params: { ...S.params } };
    if (!S.run || S.run.run_id !== match.run_id) await loadRun(match.run_id);
  } else {
    // keep showing the results that were on screen while settings are being edited
    const keep = S.run && S.run.block === b.block;
    if (!keep) S.run = null;
    const running = job && job.block === b.block && sameOptions(job.options, S.options) && sameParams(job.params, S.params);
    const probs = ruleProblems();
    const fs = firstStep();
    const note = b.runs.length && fs ? (RERUN_TEXT[fs] || RERUN_TEXT.acquire) : RERUN_TEXT.acquire;
    st.className = "status " + (running ? "running" : "missing");
    st.textContent = running ? "These settings are being computed now." : "Not computed yet for these settings. Use the bar at the bottom to run them.";
    // the run bar stays fixed at the bottom of the side panel, wherever the panel is scrolled
    const last = S.last && S.last.block === b.block ? S.last : null;
    const n = last ? Object.keys(S.config.options).filter((k) => String(S.options[k]) !== String(last.options[k])).length +
      Object.keys(S.config.params).filter((k) => Number(S.params[k]) !== Number(last.params[k])).length : 0;
    $("#run-bar-text").innerHTML = (n ? `<b>${n} setting${n > 1 ? "s" : ""} changed.</b> ` : "<b>Not computed yet.</b> ") +
      (probs.length ? `<div class="error">${probs.map(esc).join("<br>")}</div>` : esc(note)) +
      (keep ? `<div class="muted">The map and tabs still show the previous results until you run.</div>` : "");
    $("#run-discard").hidden = !last;
    bar.hidden = !!running;
    btn.disabled = probs.length > 0;
    if (!keep) clearViews();
  }
  renderBlockInfo();
}

function clearViews() {
  drawMap();
  $("#gallery").innerHTML = `<div class="empty">No results for these options yet.</div>`;
  $("#targets-table").innerHTML = "";
  $("#validation").innerHTML = `<div class="empty">No results for these options yet.</div>`;
  $("#download-groups").innerHTML = `<div class="empty">No results for these options yet.</div>`;
}

async function loadRun(runId) {
  S.run = await getJSON(`/api/runs/${runId}`);
  const known = [...S.run.layers, ...S.run.reference.map((r) => "ref:" + r.key), "none"];
  if (!known.includes(S.layer)) S.layer = S.run.layers[0] || "none";
  renderLayerRadios();
  await Promise.all([drawMap(), renderGallery(), renderTargets(), renderValidation()]);
  renderDownloads();
}

function renderLayerRadios() {
  const box = $("#layer-radios");
  const radio = (k, text) => `<label><input type="radio" name="layer" value="${k}"${k === S.layer ? " checked" : ""}> ${text}</label>`;
  box.innerHTML = S.run.layers.map((k) => radio(k, esc(S.config.layers[k]) + infoButton(k, "layer"))).join("") +
    [["stations", "Reference geophysics"], ["samples", "Reference geochemistry"]].map(([kind, title]) => {
      const refs = S.run.reference.filter((r) => r.kind === kind);
      return refs.length ? `<div class="ref-head">${title} <span>view only · not in the CPI</span></div>` +
        refs.map((r) => radio("ref:" + r.key, esc(r.label) + (r.available ? "" : ' <span class="na-tag">no data</span>') +
          infoButton("ref:" + r.key, "layer")) +
          (r.kind === "samples" && r.available && S.layer === "ref:" + r.key
            ? `<select id="geochem-el" class="el-select">${r.elements.map((e) => `<option value="${esc(e.key)}"${e.key === S.element ? " selected" : ""}>${esc(e.label)}  (${e.min} – ${e.max})</option>`).join("")}</select>` : "")).join("") : "";
    }).join("") +
    radio("none", "None");
  const sel = $("#geochem-el");
  if (sel) sel.onchange = () => { S.element = sel.value; drawRaster(); };
  box.querySelectorAll("input").forEach((i) => i.onchange = () => { S.layer = i.value; renderLayerRadios(); drawRaster(); });
}

/* ---------------- tabs ---------------- */
function renderGallery() {
  const run = S.run;
  const maps = [...run.maps.map((m) => ({ url: `/api/runs/${run.run_id}/map/${m.file}`, label: m.label, file: m.file })),
    ...run.reference.filter((r) => r.available && r.kind === "stations").map((r) => ({ url: `/api/blocks/${run.block}/reference/${r.key}/map.png`,
      file: `reference_${r.key}_map.png`, label: `Reference (not in the CPI): ${r.label}` })),
    ...run.reference.filter((r) => r.available && r.kind === "samples").flatMap((r) => ["cu", "zn", "pb", "au"]
      .filter((k) => r.elements.some((e) => e.key === k)).map((k) => ({ url: `/api/blocks/${run.block}/geochem/${k}/map.png`,
        file: `reference_stream_sediment_${k}.png`, label: `Reference (not in the CPI): ${r.elements.find((e) => e.key === k).label} in stream sediment` })))];
  $("#gallery").innerHTML = maps.map((m) => `<figure><img src="${m.url}" alt="${esc(m.label)}" loading="lazy" data-full="${m.url}">
      <figcaption><span>${esc(m.label)}</span><a href="${m.url}" download="${run.block}_${m.file}">Download PNG</a></figcaption></figure>`).join("");
  $("#gallery").querySelectorAll("img").forEach((i) => i.onclick = () => { $("#lightbox-img").src = i.dataset.full; $("#lightbox").showModal(); });
}

async function renderTargets() {
  const [zones, rows] = await Promise.all([getJSON(`/api/runs/${S.run.run_id}/zones`), getJSON(`/api/runs/${S.run.run_id}/targets`)]);
  const table = (el, cols, data, attrs) => {
    cols = cols.filter(([k]) => data.length && k in data[0]);
    const fmt = (k, v) => k === "Latitude" || k === "Longitude" ? (+v).toFixed(4) : k === "CPI_Percent" ? (+v).toFixed(1) : v;
    el.innerHTML = data.length ? `<thead><tr>${cols.map(([, h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${data.map((r) =>
      `<tr ${attrs(r)}>${cols.map(([k]) => {
        const t = k === "Priority_Tier" ? ` class="tier${String(r[k]).match(/(\d)/)?.[1]}"` : typeof r[k] === "number" ? ' class="num"' : "";
        return `<td${t}>${esc(fmt(k, r[k]))}</td>`;
      }).join("")}</tr>`).join("")}</tbody>` : "";
  };
  $("#zones-note").hidden = zones.length > 0;
  table($("#zones-table"), [["Zone_ID", "Zone"], ["Priority_Tier", "Tier"], ["Area_km2", "Area km²"], ["CPI_Max_Pct", "Peak CPI %"],
    ["CPI_Mean_Pct", "Mean CPI %"], ["Main_Evidence", "Main evidence"], ["Muscovite_Pct", "Musc %"], ["Quartz_Pct", "Qtz %"],
    ["Gossan_Pct", "Goss %"], ["Chlorite_Pct", "Chl %"], ["Lineaments_km", "Lineaments km"], ["Field_Check_Locations", "Field-check locations"],
    ["Evidence_Basis", "Based on"], ["Suggested_Follow_Up", "Suggested follow-up"]], zones, (r) => `data-zone="${esc(r.Zone_ID)}"`);
  $("#zones-table").querySelectorAll("tbody tr").forEach((tr) => tr.onclick = async () => {
    showTab("map");
    if (!$("#show-zones").checked) { $("#show-zones").checked = true; await drawVectors(); }
    const l = (G.zoneLayers || {})[tr.dataset.zone];
    if (l) { map.fitBounds(l.getBounds(), { padding: [80, 80], maxZoom: 16 }); l.openPopup(); }
  });
  table($("#targets-table"), [["Rank", "#"], ["Location_Name", "Location"], ["Target_Name", "Location"], ["Zone", "Zone"], ["Priority_Tier", "Tier"],
    ["CPI_Percent", "CPI %"], ["Muscovite_Pct", "Musc %"], ["Quartz_Pct", "Qtz %"], ["Gossan_Pct", "Goss %"], ["Chlorite_Pct", "Chl %"],
    ["Distance_to_Lineament_m", "To lineament (m)"], ["Latitude", "Lat"], ["Longitude", "Lon"], ["Check_Within_m", "Check within (m)"],
    ["Suggested_Follow_Up", "Suggested follow-up"]], rows, (r) => `data-rank="${r.Rank}" data-lat="${r.Latitude}" data-lon="${r.Longitude}"`);
  $("#targets-table").querySelectorAll("tbody tr").forEach((tr) => tr.onclick = () => {
    showTab("map");
    map.setView([+tr.dataset.lat, +tr.dataset.lon], 16);
    const m = G.targetMarkers[tr.dataset.rank];
    if (m) m.openPopup();
  });
}

async function renderValidation() {
  const run = S.run, el = $("#validation");
  if (!run.has_validation) {
    el.innerHTML = `<div class="empty">No ground truth (boreholes, lodes or explored blocks) is available for this area,
      so there is no validation. The maps, zones and field-check locations are still produced.</div>`;
    return;
  }
  const v = await getJSON(`/api/runs/${run.run_id}/validation`);
  el.innerHTML = marked.parse(v.summary) + run.validation_maps.map((m) =>
    `<img src="/api/runs/${run.run_id}/validation/${m}" alt="">`).join("");
}

function renderDownloads() {
  const run = S.run;
  const groups = {};
  run.downloads.forEach((d) => (groups[d.group] = groups[d.group] || []).push(d));
  $("#download-groups").innerHTML = Object.entries(groups).map(([g, items]) => `<div class="dgroup"><h3>${esc(g)}
    <button type="button" data-all>Select all</button></h3>${items.map((d) =>
    `<label><input type="checkbox" value="${esc(d.id)}"${/^(map|vector):/.test(d.id) || d.id === "contour:cpi" ? " checked" : ""}> ${esc(d.label)}</label>`).join("")}</div>`).join("");
  $("#download-groups").querySelectorAll("[data-all]").forEach((b) => b.onclick = () => {
    const boxes = b.closest(".dgroup").querySelectorAll("input");
    const all = [...boxes].every((x) => x.checked);
    boxes.forEach((x) => (x.checked = !all));
  });
}

function renderFormats() {
  $("#formats").innerHTML = "Vector format:" + Object.entries(S.config.formats).map(([k, v]) =>
    `<label><input type="checkbox" value="${k}"${k === "shp" ? " checked" : ""}> ${v}</label>`).join("");
}

$("#download-btn").onclick = async () => {
  if (!S.run) return;
  const items = [...document.querySelectorAll("#download-groups input:checked")].map((i) => i.value);
  const formats = [...document.querySelectorAll("#formats input:checked")].map((i) => i.value);
  if (!items.length) { alert("Select at least one item."); return; }
  const btn = $("#download-btn");
  btn.disabled = true; btn.textContent = "Preparing…";
  try {
    const r = await api(`/api/runs/${S.run.run_id}/download`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items, formats }) });
    const blob = await r.blob();
    const name = (r.headers.get("Content-Disposition") || "").match(/filename="(.+)"/)?.[1] || "geoscan.zip";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (e) { alert("Download failed: " + e.message); }
  btn.disabled = false; btn.textContent = "Download selected (.zip)";
};

function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "map") setTimeout(() => map.invalidateSize(), 50);
  if (name === "admin") renderAdmin();
}
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => { showTab(b.dataset.tab); history.replaceState(null, "", "#" + b.dataset.tab); });

/* ---------------- jobs ---------------- */
async function activeJob() {
  const js = await getJSON("/api/jobs");
  return js.reverse().find((j) => j.status === "running" || j.status === "queued") || null;
}

function showJob(j) {
  $("#job-card").hidden = false;
  $("#job-title").textContent = `${j.block} · ${j.label}`;
  $("#job-bar").style.width = `${Math.round(100 * j.progress)}%`;
  $("#job-step").textContent = j.status === "failed" ? `Failed: ${j.error}` : j.status === "done" ? "Finished." :
    j.status === "queued" ? "Waiting for another analysis to finish…" : j.step_label || "Starting…";
  $("#job-log").textContent = j.log.join("\n");
}

function pollJob(jobId) {
  clearInterval(S.jobTimer);
  S.jobTimer = setInterval(async () => {
    const j = await getJSON(`/api/jobs/${jobId}`);
    showJob(j);
    if (j.status === "done" || j.status === "failed") {
      clearInterval(S.jobTimer);
      if (j.status === "done") {
        await refreshBlocks(j.block);
        S.block = S.blocks.find((b) => b.block === j.block);
        $("#block-select").value = j.block;
        Object.assign(S.options, j.options);
        Object.assign(S.params, j.params);
        renderOptions();
        renderParams();
        await selectRun();
      }
    }
  }, 3000);
}

$("#run-discard").onclick = () => {
  if (!S.last) return;
  S.options = { ...S.last.options };
  S.params = { ...S.last.params };
  renderOptions();
  renderParams();
  selectRun();
};

$("#run-btn").onclick = async () => {
  const btn = $("#run-btn");
  if (ruleProblems().length) return;
  btn.disabled = true;
  try {
    const opts = {}, params = {};
    Object.keys(S.config.options).forEach((k) => (opts[k] = S.options[k]));
    Object.keys(S.config.params).forEach((k) => (params[k] = S.params[k]));
    const r = await (await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ block: S.block.block, options: opts, params }) })).json();
    if (r.run) { await refreshBlocks(); await selectRun(); }
    else { showJob(r.job); pollJob(r.job.job_id); await selectRun(); }
  } catch (e) { alert(e.message); }
  btn.disabled = false;
};

/* ---------------- new area ---------------- */
$("#new-area-btn").onclick = () => { $("#area-error").textContent = ""; $("#new-area").showModal(); };
$("#area-cancel").onclick = () => $("#new-area").close();
$("#new-area-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const f = $("#area-file").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("name", $("#area-name").value);
  $("#area-upload").disabled = true;
  try {
    const r = await (await api("/api/blocks", { method: "POST", body: fd })).json();
    $("#new-area").close();
    await refreshBlocks(r.block);
    S.options = defaults(S.config.options);
    S.params = defaults(S.config.params);
    renderOptions();
    renderParams();
    S.run = null;
    await selectRun();
    if (!S.block.runs.length) $("#run-status").textContent = `New area uploaded (${r.area_km2} km²). Choose options and press Run.`;
  } catch (e) { $("#area-error").textContent = e.message; }
  $("#area-upload").disabled = false;
};

$("#block-select").onchange = async () => {
  S.block = S.blocks.find((b) => b.block === $("#block-select").value);
  S.run = null;
  S.fitted = null;
  await selectRun();
};
$("#opacity").oninput = () => G.overlay && G.overlay.setOpacity(+$("#opacity").value);
["#show-contours", "#show-na"].forEach((s) => $(s).onchange = drawRaster);
["#show-targets", "#show-lineaments", "#show-zones"].forEach((s) => $(s).onchange = async () => { await drawVectors(); renderLegend(S.legend || null); });
document.querySelectorAll('input[name="base"]').forEach((r) => r.onchange = () => {
  Object.values(bases).forEach((b) => map.removeLayer(b));
  bases[r.value].addTo(map).bringToBack();
});
$("#lightbox-close").onclick = () => $("#lightbox").close();

/* ---------------- accounts and admin ---------------- */
const ACTION_TEXT = { session_start: "Signed in", upload_area: "Uploaded area", run_started: "Started analysis",
  run_finished: "Analysis finished", run_failed: "Analysis failed", open_results: "Opened results", download: "Downloaded",
  signed_out: "Signed out", user_added: "Added user", user_removed: "Removed user" };

function logDetails(e) {
  const d = [];
  if (e.file) d.push(`file <b>${esc(e.file)}</b>${e.bytes ? ` (${(e.bytes / 1024).toFixed(1)} kB)` : ""}`);
  if (e.area) d.push(`area <b>${esc(e.area)}</b>${e.area_km2 ? ` (${e.area_km2} km²)` : ""}`);
  if (e.centre_lat) d.push(`centre ${e.centre_lat}, ${e.centre_lon}`);
  if (e.result) d.push(esc(e.result));
  if (e.settings) d.push(esc(e.settings));
  if (e.changed_parameters) d.push("changed: " + Object.entries(e.changed_parameters).map(([k, v]) =>
    `${esc((S.config.params[k] || { label: k }).label)} = ${esc(v)}`).join(", "));
  if (e.recompute_from && e.action === "run_started") d.push(`recomputed from: ${esc(e.recompute_from)}`);
  if (e.results && e.action !== "open_results") d.push(`results: ${esc(e.results)}`);
  if (e.items) d.push(`${e.items.length} item(s): ${esc(e.items.join(", "))}${e.megabytes ? ` · ${e.megabytes} MB` : ""}`);
  if (e.formats && e.action === "download") d.push(`formats: ${esc(e.formats.join(", "))}`);
  if (e.error) d.push(`<span class="error">${esc(e.error)}</span>`);
  if (e.account) d.push(`account <b>${esc(e.account)}</b>${e.note ? ` (${esc(e.note)})` : ""}`);
  if (e.ip) d.push(`from ${esc(e.ip)}`);
  if (e.browser) d.push(`<span class="muted">${esc(e.browser)}</span>`);
  return d.join(" · ");
}

const localTime = (t) => t ? new Date(t).toLocaleString() : "";

async function renderAdmin() {
  const panel = $("#tab-admin");
  if (!panel.firstElementChild) {
    panel.appendChild($("#admin-template").content.cloneNode(true));
    $("#user-form").onsubmit = async (ev) => {
      ev.preventDefault();
      $("#user-error").textContent = "";
      try {
        await api("/api/admin/users", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user: $("#new-user").value, password: $("#new-pass").value, note: $("#new-note").value }) });
        $("#user-form").reset();
        renderAdmin();
      } catch (e) { $("#user-error").textContent = e.message; }
    };
    $("#log-refresh").onclick = renderAdmin;
    $("#log-user").onchange = () => drawLog();
    $("#log-action").onchange = () => drawLog();
  }
  const [users, log] = await Promise.all([getJSON("/api/admin/users"), getJSON("/api/admin/log?limit=2000")]);
  S.log = log;
  $("#users-table").innerHTML = `<thead><tr><th>User</th><th>Role</th><th>Note</th><th>Created</th><th>Last active</th><th></th></tr></thead><tbody>` +
    users.map((u) => `<tr><td><b>${esc(u.user)}</b></td><td>${esc(u.role)}</td><td>${esc(u.note)}</td>
      <td class="nowrap">${esc(localTime(u.created))}</td><td class="nowrap">${esc(localTime(u.last_active))}</td>
      <td>${u.role === "user" ? `<button class="linkbtn" data-remove="${esc(u.user)}">Remove</button>` : ""}</td></tr>`).join("") + "</tbody>";
  $("#users-table").querySelectorAll("[data-remove]").forEach((b) => b.onclick = async () => {
    if (!confirm(`Remove user "${b.dataset.remove}"? They will no longer be able to sign in.`)) return;
    await api(`/api/admin/users/${encodeURIComponent(b.dataset.remove)}`, { method: "DELETE" });
    renderAdmin();
  });
  const keep = (sel, values, all, label) => {
    const cur = sel.value;
    sel.innerHTML = `<option value="">${all}</option>` + values.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
    sel.value = values.includes(cur) ? cur : "";
  };
  keep($("#log-user"), [...new Set(log.map((e) => e.user))].sort(), "All users", (v) => v);
  keep($("#log-action"), [...new Set(log.map((e) => e.action))].sort(), "All actions", (v) => ACTION_TEXT[v] || v);
  drawLog();
}

function drawLog() {
  const u = $("#log-user").value, a = $("#log-action").value;
  const rows = (S.log || []).filter((e) => (!u || e.user === u) && (!a || e.action === a));
  $("#log-table").innerHTML = `<thead><tr><th>Time</th><th>User</th><th>Action</th><th>Details</th></tr></thead><tbody>` +
    (rows.length ? rows.map((e) => `<tr style="cursor:default"><td class="nowrap">${esc(localTime(e.time))}</td><td><b>${esc(e.user)}</b></td>
      <td class="nowrap"><span class="act ${esc(e.action)}">${esc(ACTION_TEXT[e.action] || e.action)}</span></td><td>${logDetails(e)}</td></tr>`).join("")
      : `<tr><td colspan="4" class="muted">No activity recorded yet.</td></tr>`) + "</tbody>";
}

async function signOut(ev) {
  ev.preventDefault();
  try { await api("/api/signout", { method: "POST" }); } catch (e) { /* still sign out */ }
  // Browsers keep a Basic-auth password until told otherwise: send wrong credentials on purpose so the
  // stored ones are replaced, then show a signed-out page.
  await new Promise((done) => {
    const x = new XMLHttpRequest();
    x.open("GET", "/api/logout", true, "signed-out", "signed-out");
    x.onloadend = done;
    x.send();
  });
  document.body.innerHTML = `<div class="signed-out"><h1>You are signed out</h1>
    <p>Close this tab, or <a href="/">sign in again</a>.</p>
    <p class="muted small">If the app opens without asking for a password, close all browser windows to finish signing out.</p></div>`;
}

async function initAccount() {
  S.me = await getJSON("/api/me");
  if (S.me.login) {
    $("#whoami").hidden = false;
    $("#whoami").innerHTML = `Signed in as <b>${esc(S.me.user)}</b> · activity is recorded · <a href="#" id="signout">Sign out</a>`;
    $("#signout").onclick = signOut;
  }
  if (S.me.admin) $("#admin-tab-btn").hidden = false;
}

/* ---------------- start ---------------- */
(async function init() {
  S.config = await getJSON("/api/config");
  await initAccount();
  S.options = defaults(S.config.options);
  S.params = defaults(S.config.params);
  renderOptions();
  renderParams();
  renderFormats();
  $("#params-reset").onclick = () => { S.params = defaults(S.config.params); renderParams(); selectRun(); };
  const bm = location.hash.match(/block=(\w+)/);            // links can open a given area, e.g. #block=thanewasna
  await refreshBlocks(bm ? bm[1] : undefined);
  await selectRun();
  const j = await activeJob();
  if (j) { showJob(j); pollJob(j.job_id); }
  if (location.hash.includes("params")) $("#params-box").open = true;
  // links can pre-set parameters, e.g. #params&set=targets.min_spacing_m:400
  let preset = false;
  for (const m of location.hash.matchAll(/set=([\w.]+):([\d.]+)/g)) {
    if (S.config.params[m[1]]) { S.params[m[1]] = Number(m[2]); preset = true; }
  }
  if (preset) { renderParams(); await selectRun(); }
  const lm = location.hash.match(/layer=([\w:]+)/);
  if (lm && S.run && (S.run.layers.includes(lm[1]) || refOf(lm[1]))) { S.layer = lm[1]; renderLayerRadios(); await drawRaster(); }
  const im = location.hash.match(/info=([\w.]+)/);
  if (im && (S.config.params[im[1]] || S.config.options[im[1]])) showInfo(im[1]);
  const tab = location.hash.slice(1).split("&")[0];
  if (["map", "contours", "targets", "validation", "download", "guide", "admin"].includes(tab)) showTab(tab);
})();
