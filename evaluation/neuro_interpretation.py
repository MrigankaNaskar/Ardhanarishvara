"""
Neurobiological Interpretation Module for Ardhanarishvara Phase 6.
Maps XAI-flagged regions and frequency bands to known ASD neuroscience literature.
Covers: DMN, frontal-parietal connectivity, alpha/theta abnormalities,
long-range underconnectivity, and local overconnectivity.
"""

import numpy as np

import config
from security.sanitized_logging import sanitize_errors, log_info

# ═══════════════════════════════════════════════════════════════════════════
#  ASD NEUROBIOLOGICAL BIOMARKERS (Literature-Based)
# ═══════════════════════════════════════════════════════════════════════════

ASD_FMRI_LITERATURE = {
    "Default Mode Network (DMN)": {
        "finding": "Altered DMN functional connectivity is the most replicated fMRI finding in ASD",
        "details": (
            "Hypoconnectivity between posterior cingulate cortex (PCC) and medial "
            "prefrontal cortex (mPFC) during rest. Some studies report hyperconnectivity "
            "in younger children. DMN-task positive network anticorrelation is reduced."
        ),
        "references": [
            "Padmanabhan et al. (2017) Neuropsychologia",
            "Washington et al. (2014) PNAS",
            "Uddin et al. (2013) Mol Autism",
        ],
    },
    "Frontoparietal Control": {
        "finding": "Frontal-parietal underconnectivity during executive function tasks",
        "details": (
            "Reduced functional connectivity between dorsolateral prefrontal cortex (DLPFC) "
            "and inferior parietal lobule (IPL). Associated with executive function deficits "
            "including cognitive flexibility, working memory, and planning."
        ),
        "references": [
            "Just et al. (2012) Neuron",
            "Kana et al. (2014) Brain Connect",
        ],
    },
    "Salience / Ventral Attention": {
        "finding": "Atypical salience network connectivity and interoceptive processing",
        "details": (
            "Anterior insular cortex shows altered connectivity with amygdala and ACC. "
            "Linked to social motivation and emotion processing differences in ASD."
        ),
        "references": [
            "Uddin & Menon (2009) Biol Psychiatry",
            "Green et al. (2016) JAMA Psychiatry",
        ],
    },
    "Somatomotor": {
        "finding": "Motor cortex connectivity differences linked to sensorimotor symptoms",
        "details": (
            "Primary motor and supplementary motor areas show atypical connectivity "
            "patterns. Related to motor clumsiness and repetitive motor behaviors."
        ),
        "references": [
            "Mostofsky & Ewen (2011) Curr Opin Neurol",
        ],
    },
    "Visual": {
        "finding": "Enhanced local visual processing with reduced global integration",
        "details": (
            "Visual cortex may show increased local connectivity (overconnectivity) "
            "paired with reduced long-range connections. Relates to detail-oriented "
            "perception and weak central coherence."
        ),
        "references": [
            "Samson et al. (2012) Brain Res Rev",
        ],
    },
    "Subcortical / Cerebellum": {
        "finding": "Cerebellar-cortical connectivity alterations",
        "details": (
            "Cerebellum shows reduced connectivity with sensorimotor and prefrontal "
            "regions. Thalamic relay abnormalities may affect sensory gating."
        ),
        "references": [
            "Wang et al. (2014) Brain Struct Funct",
            "Oldehinkel et al. (2019) Biol Psychiatry",
        ],
    },
}

ASD_EEG_LITERATURE = {
    "Frontal": {
        "finding": "Frontal alpha asymmetry and theta power abnormalities",
        "details": (
            "Increased frontal theta power in ASD during rest and task. "
            "Reduced left-hemispheric alpha desynchronization. "
            "Frontal theta/beta ratio elevated, linked to attentional differences."
        ),
        "references": [
            "Wang et al. (2013) BMC Psychiatry",
            "Mathewson et al. (2012) Dev Neuropsychol",
        ],
    },
    "Central": {
        "finding": "Mu rhythm suppression deficits during action observation",
        "details": (
            "Reduced mu (8-13 Hz) desynchronization over central electrodes "
            "during observation of biological motion. Linked to mirror neuron "
            "system differences and theory of mind challenges."
        ),
        "references": [
            "Oberman et al. (2005) Cognitive Brain Res",
        ],
    },
    "Parietal": {
        "finding": "Atypical alpha/beta coherence and long-range connectivity",
        "details": (
            "Reduced parietal-frontal alpha coherence suggesting long-range "
            "underconnectivity. Posterior alpha power may be increased during "
            "rest in some ASD populations."
        ),
        "references": [
            "Coben et al. (2008) BMC Med",
            "Murias et al. (2007) Biol Psychiatry",
        ],
    },
    "Temporal": {
        "finding": "Temporal cortex processing differences in social stimuli",
        "details": (
            "Superior temporal sulcus (STS) region shows altered gamma "
            "band activity during face and voice processing. Temporal "
            "coherence patterns differ in social vs. non-social contexts."
        ),
        "references": [
            "Orekhova et al. (2007) Biol Psychiatry",
        ],
    },
    "Occipital": {
        "finding": "Visual processing enhancements with altered connectivity",
        "details": (
            "Occipital gamma power may be enhanced (local overconnectivity) "
            "while occipital-frontal coherence is reduced. Related to "
            "enhanced perceptual discrimination but reduced visual integration."
        ),
        "references": [
            "Milne et al. (2009) Clin Neurophysiol",
        ],
    },
}

# General ASD connectivity patterns
ASD_GENERAL_PATTERNS = {
    "long_range_underconnectivity": {
        "finding": "Long-range underconnectivity theory",
        "details": (
            "Inter-hemispheric and long-range fronto-posterior connections are "
            "consistently reduced in ASD across both fMRI and EEG studies. "
            "This is the most robust and replicated connectivity finding."
        ),
        "references": [
            "Just et al. (2004) Brain",
            "Belmonte et al. (2004) Mol Psychiatry",
        ],
    },
    "local_overconnectivity": {
        "finding": "Local overconnectivity",
        "details": (
            "Short-range, intra-regional connections may be increased in ASD, "
            "particularly in sensory cortices. Creates an imbalance between "
            "local detail processing and global integration."
        ),
        "references": [
            "Supekar et al. (2013) Cell Reports",
        ],
    },
    "theta_beta_ratio": {
        "finding": "Elevated theta/beta ratio in ASD",
        "details": (
            "EEG theta/beta ratio is elevated in ASD, particularly at frontal "
            "sites, overlapping with ADHD-related EEG patterns. May reflect "
            "cortical hypoarousal."
        ),
        "references": [
            "Barry et al. (2003) Clin Neurophysiol",
        ],
    },
}


@sanitize_errors("Failed to interpret fMRI findings.")
def interpret_fmri_findings(fmri_importance: dict) -> list:
    """
    Match XAI-flagged fMRI networks to known ASD biomarkers.

    Args:
        fmri_importance: Output from compute_fmri_roi_importance()

    Returns:
        List of interpretation dicts matching flagged regions to literature.
    """
    network_scores = fmri_importance.get("network_scores", {})
    if not network_scores:
        return [{"network": "N/A", "finding": "No network scores available."}]

    # Threshold: networks above median importance
    threshold = np.median(list(network_scores.values()))
    hot_networks = {net: score for net, score in network_scores.items() if score >= threshold}

    interpretations = []
    for network, score in sorted(hot_networks.items(), key=lambda x: x[1], reverse=True):
        if network in ASD_FMRI_LITERATURE:
            lit = ASD_FMRI_LITERATURE[network]
            interpretations.append({
                "network": network,
                "importance_score": float(score),
                "literature_finding": lit["finding"],
                "details": lit["details"],
                "references": lit["references"],
                "status": "CONVERGES_WITH_LITERATURE",
            })
        else:
            interpretations.append({
                "network": network,
                "importance_score": float(score),
                "literature_finding": "No specific ASD literature match.",
                "details": "",
                "references": [],
                "status": "NOVEL_FINDING",
            })

    return interpretations


@sanitize_errors("Failed to interpret EEG findings.")
def interpret_eeg_findings(eeg_importance: dict) -> list:
    """
    Match XAI-flagged EEG regions to known ASD biomarkers.

    Args:
        eeg_importance: Output from compute_eeg_channel_importance()

    Returns:
        List of interpretation dicts.
    """
    region_scores = eeg_importance.get("region_scores", {})
    if not region_scores:
        return [{"region": "N/A", "finding": "No region scores available."}]

    threshold = np.median(list(region_scores.values()))
    hot_regions = {reg: score for reg, score in region_scores.items() if score >= threshold}

    interpretations = []
    for region, score in sorted(hot_regions.items(), key=lambda x: x[1], reverse=True):
        if region in ASD_EEG_LITERATURE:
            lit = ASD_EEG_LITERATURE[region]
            interpretations.append({
                "region": region,
                "importance_score": float(score),
                "literature_finding": lit["finding"],
                "details": lit["details"],
                "references": lit["references"],
                "status": "CONVERGES_WITH_LITERATURE",
            })
        else:
            interpretations.append({
                "region": region,
                "importance_score": float(score),
                "literature_finding": "No specific ASD literature match.",
                "details": "",
                "references": [],
                "status": "NOVEL_FINDING",
            })

    return interpretations


@sanitize_errors("Failed to generate neurobiological interpretation.")
def generate_neurobiological_interpretation(fmri_importance: dict,
                                             eeg_importance: dict,
                                             convergence: dict,
                                             save_path: str = None) -> str:
    """
    Generate structured neurobiological interpretation section.

    Args:
        fmri_importance: Output from compute_fmri_roi_importance()
        eeg_importance:  Output from compute_eeg_channel_importance()
        convergence:     Output from cross_modal_convergence_analysis()
        save_path:       Path to save markdown

    Returns:
        Formatted markdown string.
    """
    import os

    fmri_interp = interpret_fmri_findings(fmri_importance)
    eeg_interp = interpret_eeg_findings(eeg_importance)

    lines = [
        "# Neurobiological Interpretation\n",
        "## 1. fMRI Functional Connectivity Findings\n",
    ]

    for item in fmri_interp:
        status_emoji = "✅" if item["status"] == "CONVERGES_WITH_LITERATURE" else "🔍"
        lines.append(f"### {status_emoji} {item['network']} (score: {item['importance_score']:.3f})\n")
        lines.append(f"**Literature**: {item['literature_finding']}\n")
        if item["details"]:
            lines.append(f"{item['details']}\n")
        if item["references"]:
            lines.append("**References**:")
            for ref in item["references"]:
                lines.append(f"- {ref}")
            lines.append("")

    lines.append("\n## 2. EEG Electrophysiological Findings\n")

    for item in eeg_interp:
        status_emoji = "✅" if item["status"] == "CONVERGES_WITH_LITERATURE" else "🔍"
        key = item.get("region", item.get("network", "Unknown"))
        lines.append(f"### {status_emoji} {key} (score: {item['importance_score']:.3f})\n")
        lines.append(f"**Literature**: {item['literature_finding']}\n")
        if item["details"]:
            lines.append(f"{item['details']}\n")
        if item["references"]:
            lines.append("**References**:")
            for ref in item["references"]:
                lines.append(f"- {ref}")
            lines.append("")

    lines.append("\n## 3. Cross-Modal Convergence\n")

    convergent = convergence.get("convergent_networks", [])
    score = convergence.get("convergence_score", 0)
    lines.append(f"**Convergence Score**: {score:.2f}\n")

    if convergent:
        lines.append("**Converging Brain Networks** (flagged by both fMRI and EEG):\n")
        for net in convergent:
            lines.append(f"- **{net}**")
            if net in ASD_FMRI_LITERATURE:
                lines.append(f"  - fMRI: {ASD_FMRI_LITERATURE[net]['finding']}")
        lines.append("")
    else:
        lines.append("No convergent networks identified between modalities.\n")

    lines.append("\n## 4. General ASD Connectivity Patterns\n")
    for pattern_id, pattern in ASD_GENERAL_PATTERNS.items():
        lines.append(f"### {pattern['finding']}\n")
        lines.append(f"{pattern['details']}\n")
        lines.append("**References**:")
        for ref in pattern["references"]:
            lines.append(f"- {ref}")
        lines.append("")

    report = "\n".join(lines)

    if save_path is None:
        save_path = os.path.join(config.RESULTS_DIR, "neurobiological_interpretation.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)

    log_info(f"Saved neurobiological interpretation to {os.path.basename(save_path)}")
    return report
