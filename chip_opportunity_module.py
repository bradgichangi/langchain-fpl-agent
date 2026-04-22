from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ChipOpportunityScore:
    chip: str
    score: float  # 0–1 how good this GW is for this chip
    confidence: str  # HIGH / MEDIUM / LOW
    triggers_met: list[str]
    blockers: list[str]
    recommended_gw: int  # this GW or projected future GW
    wait_reason: str | None


class ChipTimingEngine:
    """Evaluate chip timing from squad + fixture + ranking context."""

    def __init__(self, scorer: Any | None = None):
        self.scorer = scorer

    def evaluate_all_chips(
        self,
        squad_context: dict,
        current_gw: int,
        gw_fixture_map: dict,  # gw_number -> list of fixtures
        ranked_players: list[dict],
        remaining_gws: int,
    ) -> list[ChipOpportunityScore]:
        available_chips = squad_context.get("chips_available") or []
        if not available_chips:
            available_chips = ["wildcard", "free_hit", "bench_boost", "triple_captain"]

        # Keep a predictable order before scoring sort for deterministic ties.
        canonical_order = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
        ordered = [c for c in canonical_order if c in available_chips]
        ordered.extend([c for c in available_chips if c not in ordered])

        # Ensure engine receives breakdown keys it expects.
        normalized_ranked = [self._normalize_ranked_row(p) for p in ranked_players]
        squad_context = {
            **squad_context,
            "remaining_gws": remaining_gws,
        }

        results: list[ChipOpportunityScore] = []
        for chip in ordered:
            if chip == "wildcard":
                result = self._eval_wildcard(
                    squad_context, current_gw, normalized_ranked, remaining_gws
                )
            elif chip == "free_hit":
                result = self._eval_free_hit(
                    squad_context, current_gw, gw_fixture_map, normalized_ranked
                )
            elif chip == "triple_captain":
                result = self._eval_triple_captain(squad_context, current_gw, normalized_ranked)
            elif chip == "bench_boost":
                result = self._eval_bench_boost(squad_context, current_gw, gw_fixture_map)
            else:
                # Unknown chip names are ignored instead of crashing the flow.
                continue
            results.append(result)

        return sorted(results, key=lambda x: x.score, reverse=True)

    @staticmethod
    def _normalize_ranked_row(player: dict) -> dict:
        """Map ranking rows from newer schema (`factors`) to legacy (`breakdown`)."""
        factors = player.get("breakdown") or player.get("factors") or {}
        name = player.get("web_name") or player.get("name") or "Unknown"
        fixture_count = player.get("fixtures_this_gw")
        if fixture_count is None:
            fixture_count = player.get("upcoming_fixture_count")
        if fixture_count is None:
            fixture_count = 1
        return {
            **player,
            "web_name": name,
            "breakdown": factors,
            "fixtures_this_gw": int(fixture_count),
        }

    # ------------------------------------------------------------------

    def _eval_triple_captain(self, squad_context, current_gw, ranked_players):
        triggers, blockers = [], []

        owned_ids = squad_context["squad_ids"]
        owned_ranked = [p for p in ranked_players if p["id"] in owned_ids]
        best = owned_ranked[0] if owned_ranked else None

        if not best:
            return ChipOpportunityScore(
                "triple_captain", 0.0, "LOW", [], ["No owned players ranked"], current_gw, "No data"
            )

        score_components = []

        player_score = best["score"]
        score_components.append(player_score * 0.35)
        if player_score >= 0.82:
            triggers.append(f"{best['web_name']} has elite composite score ({player_score:.2f})")
        else:
            blockers.append(f"Best owned player score only {player_score:.2f} - wait for better candidate")

        fixture_score = best["breakdown"].get("fixture", 0.0)
        score_components.append(fixture_score * 0.30)
        if fixture_score >= 0.75:
            triggers.append(f"Excellent fixture this GW (FDR score: {fixture_score:.2f})")
        else:
            blockers.append(f"Fixture not favourable enough ({fixture_score:.2f})")

        form_score = best["breakdown"].get("form", 0.0)
        score_components.append(form_score * 0.20)
        if form_score >= 0.70:
            triggers.append(f"{best['web_name']} in strong form ({form_score:.2f})")

        is_dgw = best.get("fixtures_this_gw", 1) >= 2
        dgw_bonus = 0.20 if is_dgw else 0.0
        score_components.append(dgw_bonus)
        if is_dgw:
            triggers.append("DOUBLE GAMEWEEK - TC value roughly doubled")
        else:
            blockers.append("Single fixture GW - TC value limited")

        gws_left = squad_context.get("remaining_gws", 20)
        urgency = min((38 - gws_left) / 38, 1.0)
        score_components.append(urgency * 0.05)

        total = sum(score_components)
        confidence = "HIGH" if total >= 0.75 else "MEDIUM" if total >= 0.55 else "LOW"
        wait = (
            None
            if total >= 0.72
            else f"Wait for DGW or when {best['web_name']} has FDR 1-2 home fixture"
        )

        return ChipOpportunityScore(
            chip="triple_captain",
            score=round(total, 4),
            confidence=confidence,
            triggers_met=triggers,
            blockers=blockers,
            recommended_gw=current_gw,
            wait_reason=wait,
        )

    def _eval_bench_boost(self, squad_context, current_gw, gw_fixture_map):
        triggers, blockers = [], []
        score_components = []
        bench_players = squad_context.get("bench_player_data", [])

        if not bench_players:
            return ChipOpportunityScore(
                "bench_boost", 0.0, "LOW", [], ["No bench data"], current_gw, "Need bench data"
            )

        bench_with_fixture = sum(1 for p in bench_players if p.get("has_fixture", False))
        fixture_ratio = bench_with_fixture / max(len(bench_players), 1)
        score_components.append(fixture_ratio * 0.35)
        if fixture_ratio == 1.0:
            triggers.append("All bench players have fixtures")
        elif fixture_ratio < 0.75:
            blockers.append(f"Only {bench_with_fixture}/4 bench players have fixtures")

        bench_dgw_count = sum(1 for p in bench_players if p.get("fixture_count", 1) >= 2)
        dgw_ratio = bench_dgw_count / max(len(bench_players), 1)
        score_components.append(dgw_ratio * 0.35)
        if dgw_ratio >= 0.75:
            triggers.append(f"{bench_dgw_count} bench players have double fixtures - strong BB window")
        elif dgw_ratio == 0:
            blockers.append("No bench players in a DGW - BB value low")

        bench_avg_score = sum(p.get("score", 0.4) for p in bench_players) / max(len(bench_players), 1)
        score_components.append(bench_avg_score * 0.20)
        if bench_avg_score >= 0.55:
            triggers.append(f"Bench average composite score {bench_avg_score:.2f} - quality depth")
        else:
            blockers.append(f"Bench quality low ({bench_avg_score:.2f}) - upgrade bench first")

        total_fixtures_this_gw = len(gw_fixture_map.get(current_gw, []))
        if total_fixtures_this_gw >= 12:
            score_components.append(0.10)
            triggers.append(f"High fixture volume GW ({total_fixtures_this_gw} matches)")
        else:
            blockers.append("Normal or low fixture volume GW")

        total = sum(score_components)
        confidence = "HIGH" if total >= 0.75 else "MEDIUM" if total >= 0.50 else "LOW"
        wait = None if total >= 0.70 else "Wait for a DGW where your bench has double fixtures"

        return ChipOpportunityScore(
            chip="bench_boost",
            score=round(total, 4),
            confidence=confidence,
            triggers_met=triggers,
            blockers=blockers,
            recommended_gw=current_gw,
            wait_reason=wait,
        )

    def _eval_free_hit(self, squad_context, current_gw, gw_fixture_map, ranked_players):
        triggers, blockers = [], []
        score_components = []

        owned_ids = squad_context["squad_ids"]
        blanking_owned = sum(
            1 for p in ranked_players if p["id"] in owned_ids and p.get("fixtures_this_gw", 1) == 0
        )
        blank_ratio = blanking_owned / 15
        score_components.append(blank_ratio * 0.50)
        if blanking_owned >= 6:
            triggers.append(f"{blanking_owned} of your players are blanking - FH saves the week")
        elif blanking_owned <= 2:
            blockers.append(f"Only {blanking_owned} blanks in your squad - FH not needed")

        non_blank_top = [p for p in ranked_players if p.get("fixtures_this_gw", 1) >= 1][:15]
        avg_replacement_score = sum(p["score"] for p in non_blank_top) / max(len(non_blank_top), 1)
        score_components.append(avg_replacement_score * 0.30)
        if avg_replacement_score >= 0.65:
            triggers.append("Strong pool of replacements available this GW")

        total_fixtures = len(gw_fixture_map.get(current_gw, []))
        if total_fixtures <= 6:
            score_components.append(0.20)
            triggers.append(f"Confirmed BGW ({total_fixtures} fixtures) - prime FH window")
        else:
            blockers.append("Not a blank gameweek - FH less valuable")

        total = sum(score_components)
        confidence = "HIGH" if total >= 0.75 else "MEDIUM" if total >= 0.50 else "LOW"
        wait = None if total >= 0.70 else "Hold FH for a confirmed BGW where 5+ of your players blank"

        return ChipOpportunityScore(
            chip="free_hit",
            score=round(total, 4),
            confidence=confidence,
            triggers_met=triggers,
            blockers=blockers,
            recommended_gw=current_gw,
            wait_reason=wait,
        )

    def _eval_wildcard(self, squad_context, current_gw, ranked_players, remaining_gws):
        triggers, blockers = [], []
        score_components = []

        owned_ids = squad_context["squad_ids"]
        owned_players = [p for p in ranked_players if p["id"] in owned_ids]
        all_players = ranked_players

        avg_owned_score = sum(p["score"] for p in owned_players) / max(len(owned_players), 1)
        top15_score = sum(p["score"] for p in all_players[:15]) / 15
        quality_gap = max(top15_score - avg_owned_score, 0)
        score_components.append(min(quality_gap * 3, 0.40))
        if quality_gap >= 0.10:
            triggers.append(f"Squad quality gap of {quality_gap:.2f} vs optimal 15 - WC needed")
        else:
            blockers.append(f"Squad quality gap only {quality_gap:.2f} - WC not urgent")

        injury_count = sum(
            1 for p in owned_players if p.get("breakdown", {}).get("availability", 1.0) < 0.5
        )
        score_components.append(min(injury_count / 4, 0.25))
        if injury_count >= 3:
            triggers.append(f"{injury_count} players unavailable - squad in crisis")

        owned_avg_fixture = sum(p.get("breakdown", {}).get("fixture", 0.5) for p in owned_players) / max(
            len(owned_players), 1
        )
        score_components.append((1 - owned_avg_fixture) * 0.20)
        if owned_avg_fixture < 0.45:
            triggers.append("Squad has poor upcoming fixtures - ideal WC timing")

        urgency = min(current_gw / 38, 1.0)
        score_components.append(urgency * 0.15)
        if current_gw >= 30:
            triggers.append("Late season - limited GWs to benefit from WC")

        total = sum(score_components)
        confidence = "HIGH" if total >= 0.70 else "MEDIUM" if total >= 0.45 else "LOW"
        wait = None if total >= 0.65 else "Hold WC - squad quality gap not significant enough yet"

        return ChipOpportunityScore(
            chip="wildcard",
            score=round(total, 4),
            confidence=confidence,
            triggers_met=triggers,
            blockers=blockers,
            recommended_gw=current_gw,
            wait_reason=wait,
        )


def serialize_chip_scores(scores: list[ChipOpportunityScore]) -> list[dict]:
    return [asdict(s) for s in scores]