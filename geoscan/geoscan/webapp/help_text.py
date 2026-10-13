"""Plain-language explanations shown by the (i) buttons in the web app.

Each entry: what (what it is), how (how the method uses it, formula), up / down (effect of raising /
lowering it), why (why the default), evidence (what validation says, optional). Keep these in step with
the methodology document.
"""

INFO = {
    # ------------------------------------------------------------------ method options
    "structure.dem_source": {
        "what": "The elevation model (terrain map) that lineaments are traced from.",
        "how": "Hillshades from 8 sun directions and valley/ridge axes are computed from this elevation model, "
               "then straight edges are found and joined into lines.",
        "choices": "ALOS AW3D30 (JAXA, from stereo optical images, 2006-2011). "
                   "SRTM (NASA radar, February 2000; used in the 24 September version). "
                   "Copernicus GLO-30 (TanDEM-X radar, 2011-2015).",
        "why": "AW3D30 was the most accurate model at the 47 Thanewasna borehole collars (1.4 m error, against "
               "3.1 m for SRTM). It also gave the best results at both blocks when the complete map was rebuilt.",
        "evidence": "Thanewasna, structure layer against Cu lodes: AUC 0.77 with AW3D30, 0.71 with SRTM. "
                    "Against boreholes: 0.85 and 0.72.",
    },
    "structure.canny_tile_m": {
        "what": "How the 'is this a strong edge?' threshold is set when lines are traced.",
        "how": "An edge counts when its contrast is in the top 20 % (to extend a line) or top 5 % (to start one). "
               "'Whole area' ranks every pixel of the area together. 'Per 3 km tile' ranks each 3 km square on its own.",
        "up": "Per 3 km tile: flat ground gets its own fair share of edges, so subtle structures in plains are found.",
        "down": "Whole area: hilly parts use up the top-5 % edges and flat parts get almost none.",
        "why": "Per-tile thresholds found more of the ore-parallel structure at Thanewasna. At Kehalpur they also "
               "found more lines parallel to the mapped foliation.",
    },
    # ------------------------------------------------------------------ CPI and targets
    "targets.w_muscovite": {
        "what": "How much muscovite / sericite (phyllic alteration) counts in the hydrothermal score H.",
        "how": "H = w_musc·muscovite + w_qtz·quartz + w_goss·gossan + w_chl·chlorite, each layer 0-1. "
               "The four weights must add up to 1.",
        "up": "Areas with strong white mica rise in the ranking.",
        "down": "White-mica areas count less; the other three minerals decide more.",
        "why": "Expert-assigned, following porphyry-style zoning. The sericite (phyllic) zone sits closest to the "
               "ore, so it gets the largest weight. This is a judgement, not a measured value.",
        "evidence": "Thanewasna: muscovite layer AUC 0.62 at lodes, 0.76 at boreholes.",
    },
    "targets.w_quartz": {
        "what": "How much quartz / silicification counts in the hydrothermal score H.",
        "how": "H = w_musc·muscovite + w_qtz·quartz + w_goss·gossan + w_chl·chlorite, each layer 0-1. "
               "The four weights must add up to 1.",
        "up": "Silica-rich areas, from ASTER thermal data at 90 m, rise in the ranking.",
        "down": "Quartz counts less.",
        "why": "Expert-assigned; silicification often accompanies the phyllic zone.",
        "evidence": "Thanewasna: the quartz layer shows no skill at the lodes (AUC 0.51) and some at boreholes (0.68). "
                    "The ASTER pixels (90 m) are coarse for a 2 km block.",
    },
    "targets.w_gossan": {
        "what": "How much gossan / iron oxide counts in the hydrothermal score H.",
        "how": "H = w_musc·muscovite + w_qtz·quartz + w_goss·gossan + w_chl·chlorite, each layer 0-1. "
               "The four weights must add up to 1.",
        "up": "Iron-stained, oxidised ground rises in the ranking.",
        "down": "Iron oxide counts less. Useful where laterite or ferruginous soil (not gossan) is widespread.",
        "why": "Expert-assigned. A gossan is the weathered cap of sulphides.",
        "evidence": "Thanewasna: the strongest single layer (AUC 0.77 at lodes, 0.83 at boreholes). Two independent "
                    "satellites (ASTER, Sentinel-2) show the same iron-oxide pattern.",
    },
    "targets.w_chlorite": {
        "what": "How much chlorite / epidote (propylitic alteration) counts in the hydrothermal score H.",
        "how": "H = w_musc·muscovite + w_qtz·quartz + w_goss·gossan + w_chl·chlorite, each layer 0-1. "
               "The four weights must add up to 1.",
        "up": "Chlorite-rich areas rise in the ranking.",
        "down": "Chlorite counts less.",
        "why": "Expert-assigned and smallest, because the propylitic zone is the outer halo, not the ore.",
        "evidence": "Thanewasna: chlorite is lower at lode-intersecting boreholes than at halo boreholes (AUC 0.29). "
                    "In this deposit it behaves as a distal indicator, so a lower weight is worth testing.",
    },
    "targets.w_structure": {
        "what": "How much the nearness to a lineament counts in the final score, compared with the minerals.",
        "how": "CPI = (1 − s)·H + s·S_prox, where S_prox = exp(−distance to nearest lineament / λ).",
        "up": "Zones are pulled towards mapped lineaments. Areas with no mapped line score lower.",
        "down": "The minerals decide almost alone.",
        "why": "Expert-assigned: alteration is the main evidence (70 %) and structure supports it (30 %).",
        "evidence": "Thanewasna: structure layer alone AUC 0.77 at lodes, 0.85 at boreholes. Remember that "
                    "lineaments come from terrain and miss buried structures.",
    },
    "targets.structure_lambda_m": {
        "what": "How quickly the benefit of being near a lineament fades with distance.",
        "how": "S_prox = exp(−d / λ). At d = λ the structural score is 37 %, at 2λ it is 14 %, at 3λ 5 %.",
        "up": "Lineaments influence a wider strip. The map gets smoother and more ground benefits.",
        "down": "Only ground right next to a lineament benefits.",
        "why": "Assumed width of fracture-controlled alteration around a structure (a few hundred metres).",
        "evidence": "Thanewasna: λ = 400 m gave lode AUC 0.74, against 0.73 with 250 m. The difference is small.",
    },
    "targets.tier1": {
        "what": "Score a zone must reach somewhere inside it to be a Tier 1 (high-priority) zone.",
        "how": "A prospective zone is Tier 1 when its peak CPI >= this value. It is also the highest contour on the "
               "CPI map. CPI is not re-stretched, so these are absolute values.",
        "up": "Fewer zones are called Tier 1.",
        "down": "More zones are called Tier 1.",
        "why": "Expert-assigned class boundary. It changes labels and suggested follow-up, not the zone outlines.",
        "evidence": "Thanewasna: the Tier 1 zone covers 17.5 % of the block and holds 53 % of the known lode area "
                    "(3.1 times chance) and 6 of the 8 boreholes that intersect a lode.",
    },
    "targets.tier2": {
        "what": "The outline of a prospective zone: all connected ground with CPI at or above this value.",
        "how": "Zones are the connected areas with CPI >= this value. A zone is Tier 1 if it reaches the Tier 1 "
               "threshold somewhere, otherwise Tier 2. This is the middle contour on the CPI map.",
        "up": "Smaller, tighter zones: less ground to check, but more of the mineralisation is left outside.",
        "down": "Larger zones that hold more of the mineralisation but give less focus. Very low values merge "
                "whole districts into one zone.",
        "why": "Expert-assigned. Tested against the Tier 3 line as the boundary, which was too loose.",
        "evidence": "Thanewasna, share of block covered -> share of lode area inside: CPI >= 28 %: 56 % -> 74 % "
                    "(1.3 times chance); CPI >= 50 %: 16.5 % -> 48.5 % (2.9 times); CPI >= 70 %: 7 % -> 35 % (5 times).",
    },
    "targets.tier3": {
        "what": "The outer halo around the zones, and the score an isolated weak anomaly needs to be listed as a Tier 3 zone.",
        "how": "Ground with CPI >= this value around a Tier 1 / 2 zone is its outer halo (lowest contour on the map). "
               "An anomaly that reaches this value but never the Tier 2 threshold, and is separate from every "
               "Tier 1 / 2 zone, is listed as a Tier 3 zone.",
        "up": "Narrower halos and fewer Tier 3 zones.", "down": "Wider halos and more weak Tier 3 zones.",
        "why": "Expert-assigned class boundary.",
    },
    "targets.min_cpi": {
        "what": "No field-check location is suggested where the score is below this value.",
        "how": "Only cells with CPI >= this value are considered when the suggested field-check locations are picked.",
        "up": "Fewer locations, all in strong anomalies.",
        "down": "More locations, reaching into weak anomalies (still at most the maximum number).",
        "why": "Expert-assigned, just below the Tier 3 boundary.",
    },
    "targets.min_spacing_m": {
        "what": "The minimum distance between two suggested field-check locations.",
        "how": "Cells are taken from the highest score down. A cell closer than this to an already chosen location is "
               "skipped: it is the weaker neighbour of that location, part of the same anomaly. The same distance is "
               "given as the 'check within' radius of each location.",
        "up": "Fewer, more spread-out locations: roughly one per anomaly.",
        "down": "Several locations can sit on one large zone.",
        "why": "Assumed: about the size of one anomaly, so one zone does not take all the top ranks. The locations are "
               "starting points for field work. Their exact position is not significant, and they are not drill collars.",
        "evidence": "Thanewasna: only 3 of the top 10 locations lie within 100 m of a lode (1.9 expected by chance). "
                    "The method predicts zones far better than points.",
    },
    "targets.n_targets": {
        "what": "The largest number of suggested field-check locations listed.",
        "how": "Picking stops when this many locations are found (or when no cell above the minimum score is left).",
        "up": "A longer list.", "down": "A shorter list.",
        "why": "A practical list length for planning field traverses.",
    },
    # ------------------------------------------------------------------ alteration layers
    "alteration.sigma_core_m": {
        "what": "Size of the sharp 'hot spot' drawn around each EMIT detection.",
        "how": "Every detection (a 60 m pixel) is spread with a Gaussian of this σ. The core keeps the peak where "
               "the mineral was actually seen.",
        "up": "Hot spots are blurred and nearby detections merge.",
        "down": "Hot spots stay pixel-sharp and spotty.",
        "why": "Expert choice: about 1.5 EMIT pixels, enough to remove pixel noise but keep detail.",
    },
    "alteration.sigma_halo_m": {
        "what": "Size of the wider zone drawn around each detection, standing for the alteration halo.",
        "how": "A second, wider Gaussian of this σ is combined with the core. It decides how far a zone reaches "
               "beyond the detected pixels.",
        "up": "Wider, more continuous zones; isolated detections spread further.",
        "down": "Tighter zones that hug the detections.",
        "why": "Expert choice: alteration halos around a mineralised system are typically a few hundred metres wide.",
    },
    "alteration.theta_decay": {
        "what": "How many detections must be close together before a halo is believed.",
        "how": "confidence = 1 − exp(−W / θ), where W is the share of detections within the halo around a pixel.",
        "up": "A single lonely detection fades away (treated as noise); only clusters show strongly.",
        "down": "Even one detection shows at full strength.",
        "why": "Expert choice.",
    },
    "alteration.reference_percentile": {
        "what": "Which value is shown as 100 % on the mineral maps.",
        "how": "Intensity = field / (value of the field at this percentile inside the area), capped at 100 %.",
        "up": "Fewer pixels reach 100 %; the strongest areas stand out more.",
        "down": "More of the map looks saturated at 100 %.",
        "why": "The 98th percentile ignores the most extreme 2 % of pixels. Percentages are relative to "
               "this area, so 80 % here is not comparable with 80 % in another area.",
    },
    "alteration.sericite_min_nm": {
        "what": "Shortest wavelength of the Al-OH absorption that is accepted as white mica (muscovite / sericite).",
        "how": "Muscovite has its main absorption near 2200 nm. Detections whose absorption minimum falls outside "
               "the window are not counted as muscovite.",
        "up": "Excludes sodic micas (paragonite, near 2190 nm).",
        "down": "Accepts more sodic micas.",
        "why": "From spectral-library positions of the muscovite family. Kaolinite and smectite absorb at similar "
               "positions, so the window cannot exclude them completely.",
    },
    "alteration.sericite_max_nm": {
        "what": "Longest wavelength of the Al-OH absorption that is accepted as white mica.",
        "how": "Muscovite has its main absorption near 2200 nm. Detections whose absorption minimum falls outside "
               "the window are not counted as muscovite.",
        "up": "Accepts phengitic (Fe/Mg-rich) micas, which absorb at 2215-2225 nm.",
        "down": "Keeps only classic muscovite / sericite.",
        "why": "From spectral-library positions of the muscovite family.",
    },
    "alteration.quartz_qi_weight": {
        "what": "Balance between the two ASTER thermal indices in the quartz layer.",
        "how": "QSI = w·QI + (1 − w)·SI. QI = B11² / (B10·B12) is the published quartz index (Ninomiya 2003). "
               "SI = B13 / B12 is a silica ratio inherited from earlier work with no verified primary source.",
        "up": "Relies more on the published quartz index.",
        "down": "Relies more on the unverified silica ratio.",
        "why": "Expert choice. This weighting is one of the review points in the methodology.",
    },
    "alteration.quartz_background_percentile": {
        "what": "How much of the weakest quartz signal is treated as ordinary rock (0 %).",
        "how": "Quartz intensity = (QSI − P_background) / (P98 − P_background), clipped to 0-100 %.",
        "up": "Only the strongest quartz shows; more of the area becomes 0 %.",
        "down": "More of the area shows some quartz.",
        "why": "Project assumption: the lowest 15 % of the area is background.",
    },
    # ------------------------------------------------------------------ lineaments
    "structure.min_length_m": {
        "what": "Lineaments shorter than this are dropped.",
        "how": "After the line pieces are joined, any line shorter than this is removed.",
        "up": "Only major structures remain.",
        "down": "More short lines, including drainage and noise.",
        "why": "Expert choice: a structure that guides fluids over a deposit is usually at least ~1 km long.",
    },
    "structure.min_segment_m": {
        "what": "Shortest straight piece the line finder will accept before joining pieces.",
        "how": "A Hough transform finds straight pieces on the edge map; pieces shorter than this are ignored.",
        "up": "Fewer, cleaner pieces.", "down": "More fragments.",
        "why": "Project assumption.",
    },
    "structure.max_gap_m": {
        "what": "Largest break that is bridged inside one straight piece.",
        "how": "Edges along the same line with a gap up to this size are treated as one piece.",
        "up": "Broken lines are joined more easily.", "down": "Lines break at small gaps.",
        "why": "Project assumption (3 pixels at 30 m).",
    },
    "structure.canny_low_quantile": {
        "what": "How weak an edge may be and still extend a line.",
        "how": "Edges in the top (1 − value) of contrast can continue a line that a strong edge started.",
        "up": "Fewer, shorter lines.", "down": "Lines run further through weak edges.",
        "why": "Project assumption (top 20 %).",
    },
    "structure.canny_high_quantile": {
        "what": "How strong an edge must be to start a line.",
        "how": "Only edges in the top (1 − value) of contrast can start a line.",
        "up": "Only the sharpest features start lines.", "down": "More lines start.",
        "why": "Project assumption (top 5 %).",
    },
    "structure.scarp_min_step_m": {
        "what": "When a line is called a scarp (a cliff or plateau edge) instead of a valley or ridge.",
        "how": "The terrain is compared 90 m either side of the line. If one side is consistently higher by at "
               "least this much, the line is a scarp.",
        "up": "Fewer lines are called scarps.", "down": "More lines are called scarps.",
        "why": "Expert choice. Erosional scarps (such as the Deccan plateau edge) are not fluid pathways.",
    },
    "structure.exclude_scarps": {
        "what": "Whether scarps are left out of the CPI.",
        "how": "Scarps always stay in the lineament map and vectors. When this is on, they are not used for the "
               "structural score or density.",
        "up": "On: cliff edges do not attract targets.", "down": "Off: every line counts.",
        "why": "Project assumption, checked at both blocks.",
    },
    # ------------------------------------------------------------------ EMIT detection
    "spectral.detect_sigma": {
        "what": "How sure we must be that an absorption is real before calling it a detection.",
        "how": "Detected when band depth > k × its uncertainty. The uncertainty comes from EMIT's own per-band "
               "error, propagated with 64 Monte-Carlo draws. k = 2 is about 95 % confidence.",
        "up": "Fewer, more certain detections.", "down": "More, less certain detections.",
        "why": "Project assumption: the usual 2σ significance level.",
    },
    "spectral.min_depth": {
        "what": "Weakest absorption that can count as a detection, however precise it is.",
        "how": "Detected only when band depth is also above this value (0.01 = a 1 % dip).",
        "up": "Shallow absorptions are ignored.", "down": "Very shallow dips can count.",
        "why": "Project assumption.",
    },
    "spectral.min_albedo": {
        "what": "Pixels darker than this are skipped.",
        "how": "Pixels with reflectance below this value are masked before band depths are measured.",
        "up": "More dark ground (shadow, water, burnt areas) is skipped.", "down": "Darker pixels are used.",
        "why": "Project assumption: band depths are unreliable on very dark surfaces.",
    },
}

LAYER_INFO = {
    "cpi": "Copper Prospectivity Index: (1 − s) × hydrothermal score + s × nearness to lineaments. "
           "0-100 %; contours at the Tier 3 / 2 / 1 thresholds. The Tier 2 contour is the outline of the "
           "prospective zones.",
    "muscovite": "White mica (sericite, phyllic alteration) from the EMIT Al-OH absorption near 2200 nm. "
                 "Relative intensity: 100 % = strongest 2 % of this area.",
    "chlorite": "Chlorite / epidote (propylitic alteration) from the EMIT Fe-OH (2250 nm) and Mg-OH (2330 nm) "
                "absorptions, with carbonate-dominated pixels excluded using EMIT L2B.",
    "gossan": "Iron oxide (Fe³⁺) from the EMIT 900 nm absorption. It cannot separate gossan from laterite or "
              "ferruginous soil.",
    "quartz": "Quartz / silicification from ASTER thermal emissivity (90 m): QSI = w·QI + (1 − w)·SI.",
    "lineament_density": "Length of lineaments (km) per km² within 1 km.",
}
