"""Catalog of DTU Health Tech services relevant to probing tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName, TaskLevel


DtuServiceCategory = Literal["ptm", "structure", "sorting", "immunology", "dataset"]


@dataclass(frozen=True)
class DtuServiceSpec:
    """Metadata for a DTU service that can inspire or provide probing data."""

    name: str
    category: DtuServiceCategory
    level: TaskLevel
    objective: ObjectiveName
    target: str
    url: str
    description: str
    dataset_url: str | None = None
    notes: str = ""


DTU_SERVICES_BASE_URL = "https://services.healthtech.dtu.dk"

DTU_PROBING_CATALOG: Dict[str, DtuServiceSpec] = {
    "dictyoglyc": DtuServiceSpec(
        name="dictyoglyc",
        category="ptm",
        level="residue",
        objective="binary",
        target="o_glcnac_glycosylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/DictyOGlyc-1.1/",
        description="O-alpha-GlcNAc glycosylation site prediction.",
    ),
    "netcglyc": DtuServiceSpec(
        name="netcglyc",
        category="ptm",
        level="residue",
        objective="binary",
        target="c_mannosylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetCGlyc-1.0/",
        description="C-mannosylation site prediction in mammalian proteins.",
    ),
    "netnglyc": DtuServiceSpec(
        name="netnglyc",
        category="ptm",
        level="residue",
        objective="binary",
        target="n_linked_glycosylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetNGlyc-1.0/",
        description="N-linked glycosylation site prediction.",
    ),
    "netoglyc": DtuServiceSpec(
        name="netoglyc",
        category="ptm",
        level="residue",
        objective="binary",
        target="o_galnac_glycosylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetOGlyc-4.0/",
        description="O-GalNAc glycosylation site prediction.",
        dataset_url=f"{DTU_SERVICES_BASE_URL}/services/OglycBase/",
    ),
    "netphos": DtuServiceSpec(
        name="netphos",
        category="ptm",
        level="residue",
        objective="binary",
        target="phosphorylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetPhos-3.1/",
        description="Generic eukaryotic phosphorylation site prediction.",
    ),
    "netphosbac": DtuServiceSpec(
        name="netphosbac",
        category="ptm",
        level="residue",
        objective="binary",
        target="bacterial_phosphorylation_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetPhosBac-1.0/",
        description="Generic bacterial phosphorylation site prediction.",
    ),
    "prop": DtuServiceSpec(
        name="prop",
        category="ptm",
        level="residue",
        objective="binary",
        target="propeptide_cleavage_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/ProP-1.0/",
        description="Arginine and lysine propeptide cleavage site prediction.",
    ),
    "deeptmhmm": DtuServiceSpec(
        name="deeptmhmm",
        category="structure",
        level="residue",
        objective="multiclass",
        target="transmembrane_topology",
        url=f"{DTU_SERVICES_BASE_URL}/services/DeepTMHMM-1.0/",
        description="Transmembrane helix and topology prediction.",
    ),
    "netsurfp": DtuServiceSpec(
        name="netsurfp",
        category="structure",
        level="residue",
        objective="multiclass",
        target="secondary_structure_accessibility",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetSurfP-3.0/",
        description="Secondary structure and relative solvent accessibility prediction.",
    ),
    "netturnp": DtuServiceSpec(
        name="netturnp",
        category="structure",
        level="residue",
        objective="multiclass",
        target="beta_turn_type",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetTurnP-1.0/",
        description="Beta-turn residue and beta-turn type prediction.",
    ),
    "tmhmm": DtuServiceSpec(
        name="tmhmm",
        category="structure",
        level="residue",
        objective="multiclass",
        target="transmembrane_topology",
        url=f"{DTU_SERVICES_BASE_URL}/services/TMHMM-2.0/",
        description="Transmembrane helix prediction.",
    ),
    "netsolp": DtuServiceSpec(
        name="netsolp",
        category="structure",
        level="protein",
        objective="binary",
        target="ecoli_soluble_expression",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetSolP-1.0/",
        description="Solubility and usability prediction for proteins expressed in E. coli.",
    ),
    "deeploc": DtuServiceSpec(
        name="deeploc",
        category="sorting",
        level="protein",
        objective="multiclass",
        target="subcellular_localization",
        url=f"{DTU_SERVICES_BASE_URL}/services/DeepLoc-2.1/",
        description="Eukaryotic protein subcellular localization prediction.",
    ),
    "deeplocpro": DtuServiceSpec(
        name="deeplocpro",
        category="sorting",
        level="protein",
        objective="multiclass",
        target="prokaryotic_subcellular_localization",
        url=f"{DTU_SERVICES_BASE_URL}/services/DeepLocPro-1.0/",
        description="Prokaryotic protein subcellular localization prediction.",
    ),
    "lipop": DtuServiceSpec(
        name="lipop",
        category="sorting",
        level="residue",
        objective="multiclass",
        target="signal_peptidase_cleavage_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/LipoP-1.0/",
        description="Signal peptidase I and II cleavage site prediction in Gram-negative bacteria.",
    ),
    "netnes": DtuServiceSpec(
        name="netnes",
        category="sorting",
        level="residue",
        objective="binary",
        target="nuclear_export_signal",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetNES-1.1/",
        description="Leucine-rich nuclear export signal prediction.",
        dataset_url=f"{DTU_SERVICES_BASE_URL}/services/NESbase/",
    ),
    "signalp": DtuServiceSpec(
        name="signalp",
        category="sorting",
        level="residue",
        objective="multiclass",
        target="signal_peptide_cleavage_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/SignalP-6.0/",
        description="Signal peptide and cleavage-site prediction across domains of life.",
        dataset_url=f"{DTU_SERVICES_BASE_URL}/services/SignalP-5.0/datasets.php",
    ),
    "targetp": DtuServiceSpec(
        name="targetp",
        category="sorting",
        level="protein",
        objective="multiclass",
        target="organelle_targeting",
        url=f"{DTU_SERVICES_BASE_URL}/services/TargetP-2.0/",
        description="Mitochondrial, chloroplastic, secretory pathway, or other localization prediction.",
    ),
    "tatp": DtuServiceSpec(
        name="tatp",
        category="sorting",
        level="residue",
        objective="binary",
        target="tat_signal_peptide_cleavage_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/TatP-1.0/",
        description="Twin-arginine signal peptide presence and cleavage-site prediction.",
    ),
    "bepipred": DtuServiceSpec(
        name="bepipred",
        category="immunology",
        level="residue",
        objective="binary",
        target="b_cell_epitope",
        url=f"{DTU_SERVICES_BASE_URL}/services/BepiPred-3.0/",
        description="Linear B-cell epitope prediction from protein sequence.",
    ),
    "netmhcpan": DtuServiceSpec(
        name="netmhcpan",
        category="immunology",
        level="protein",
        objective="regression",
        target="mhc_i_binding_affinity",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetMHCpan-4.1/",
        description="Peptide-MHC class I binding prediction.",
        notes="This is a peptide/HLA-pair task rather than a simple single-protein task.",
    ),
    "netmhciipan": DtuServiceSpec(
        name="netmhciipan",
        category="immunology",
        level="protein",
        objective="regression",
        target="mhc_ii_binding_affinity",
        url=f"{DTU_SERVICES_BASE_URL}/services/NetMHCIIpan-4.3/",
        description="Peptide-MHC class II binding prediction.",
        notes="This is a peptide/HLA-pair task rather than a simple single-protein task.",
    ),
    "dna2protss": DtuServiceSpec(
        name="dna2protss",
        category="dataset",
        level="residue",
        objective="multiclass",
        target="secondary_structure",
        url=f"{DTU_SERVICES_BASE_URL}/services/DNA2protSS/",
        description="mRNA sequences and corresponding protein secondary-structure assignments.",
        dataset_url=f"{DTU_SERVICES_BASE_URL}/services/DNA2protSS/",
    ),
    "signalp_datasets": DtuServiceSpec(
        name="signalp_datasets",
        category="dataset",
        level="residue",
        objective="multiclass",
        target="signal_peptide_cleavage_site",
        url=f"{DTU_SERVICES_BASE_URL}/services/SignalP-5.0/datasets.php",
        description="Signal peptide datasets listed by DTU.",
        dataset_url=f"{DTU_SERVICES_BASE_URL}/services/SignalP-5.0/datasets.php",
    ),
}


def get_dtu_service(name: str) -> DtuServiceSpec:
    """Return one registered DTU service specification."""
    key = str(name).strip().lower().replace("-", "_")
    spec = DTU_PROBING_CATALOG.get(key)
    if spec is None:
        supported = ", ".join(sorted(DTU_PROBING_CATALOG))
        raise EmbeddingInputError(f"Unknown DTU service {name!r}. Supported values: {supported}.")
    return spec


def list_dtu_services(
    *,
    category: DtuServiceCategory | None = None,
    level: TaskLevel | None = None,
    has_dataset: bool | None = None,
) -> List[DtuServiceSpec]:
    """Return registered DTU service specifications."""
    values = list(DTU_PROBING_CATALOG.values())
    if category is not None:
        values = [value for value in values if value.category == category]
    if level is not None:
        values = [value for value in values if value.level == level]
    if has_dataset is not None:
        values = [value for value in values if (value.dataset_url is not None) == has_dataset]
    return values


__all__ = [
    "DTU_PROBING_CATALOG",
    "DTU_SERVICES_BASE_URL",
    "DtuServiceCategory",
    "DtuServiceSpec",
    "get_dtu_service",
    "list_dtu_services",
]
