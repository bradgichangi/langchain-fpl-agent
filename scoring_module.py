"""
FPL Player Scoring Module
=========================
A composite scoring engine for Fantasy Premier League player recommendations.
Designed for integration with Agentverse agents using the Hermes LLM.

Usage:
    from fpl_scoring_module import FPLScorer

    scorer = FPLScorer()
    score, breakdown = scorer.score_player(player_data, fixtures, sentiment_score)
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class Position(Enum):
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4


# Default weights per factor — sum to 1.0
# These are tuned per position further below
BASE_WEIGHTS = {
    "fixture":       0.22,
    "form":          0.18,
    "value":         0.08,
    "home_away":     0.07,
    "xg_xa":         0.10,
    "availability":  0.12,
    "momentum":      0.08,
    "clean_sheet":   0.05,
    "bonus_rate":    0.05,
    "set_piece":     0.05,
}

# Position-specific weight overrides
POSITION_WEIGHTS = {
    # Tuned to FPL scoring rules: the dominant point sources per position are
    # reflected in the dominant weights. Dead weights (xG for GKs, CS for FWDs)
    # are zeroed so they don't quietly drag scores down.
    Position.GOALKEEPER: {
        "fixture":       0.25,   # drives CS probability and goals-conceded risk
        "form":          0.10,   # largely proxies CS/saves — trimmed to avoid double-counting
        "value":         0.07,
        "home_away":     0.08,   # home GKs concede less → higher CS odds
        "xg_xa":         0.00,   # GKs effectively never register xG/xA
        "availability":  0.15,   # benched starter = zero floor; biggest disaster risk
        "momentum":      0.05,   # transfer noise rarely signals GK quality
        "clean_sheet":   0.23,   # primary point source after appearances
        "bonus_rate":    0.07,   # saves + CS drive BPS
        "set_piece":     0.00,   # GKs effectively never take set pieces
    },
    Position.DEFENDER: {
        "fixture":       0.18,   # CS factor already embeds fixture quality
        "form":          0.15,
        "value":         0.08,
        "home_away":     0.05,
        "xg_xa":         0.12,   # attacking fullbacks are the premium-defender meta
        "availability":  0.12,
        "momentum":      0.06,
        "clean_sheet":   0.15,   # 4 pts × CS rate materially shifts rankings
        "bonus_rate":    0.05,   # defender BPS overlaps with CS signal
        "set_piece":     0.04,   # Trippier/TAA-types — modest but real assist signal
    },
    Position.MIDFIELDER: {
        "fixture":       0.15,   # attacking mids less fixture-correlated than defenders
        "form":          0.18,   # form tracks attacking output well
        "value":         0.08,
        "home_away":     0.06,
        "xg_xa":         0.18,   # primary predictor of goals/assists
        "availability":  0.10,
        "momentum":      0.08,
        "clean_sheet":   0.02,   # only 1 pt — near-negligible
        "bonus_rate":    0.07,   # BPS from attacking contributions
        "set_piece":     0.08,   # primary pen/FK takers (Saka, Palmer) are massive
    },
    Position.FORWARD: {
        "fixture":       0.12,   # xG/xA already captures attacking fixture quality
        "form":          0.18,
        "value":         0.08,   # budget FWDs (Beto, Wissa) are a real edge
        "home_away":     0.06,
        "xg_xa":         0.26,   # dominant predictor of forward returns
        "availability":  0.10,
        "momentum":      0.07,
        "clean_sheet":   0.00,   # forwards receive no CS points
        "bonus_rate":    0.05,
        "set_piece":     0.08,   # primary pen takers (Haaland, Watkins) command premium
    },
}

# Gameweek fixture decay weights (closer fixtures matter more)
FIXTURE_DECAY = [0.35, 0.25, 0.18, 0.13, 0.09]

# Score tier labels — absolute cutoffs used for single-player scoring.
# In practice most factor ceilings cap composite scores below ~0.65, so prefer
# `TIER_PERCENTILES` via `apply_percentile_tiers()` when scoring a population.
TIERS = [
    (0.85, "⭐ MUST START"),
    (0.72, "✅ STRONG PICK"),
    (0.58, "🟡 VIABLE OPTION"),
    (0.42, "🟠 RISKY PICK"),
    (0.00, "🔴 AVOID"),
]

# Percentile-based tier bands (top X% of a ranked population).
# Tuned so roughly the top ~2% of the player pool are MUST-START tier.
TIER_PERCENTILES = [
    (98, "⭐ MUST START"),
    (90, "✅ STRONG PICK"),
    (75, "🟡 VIABLE OPTION"),
    (40, "🟠 RISKY PICK"),
    (0,  "🔴 AVOID"),
]


def tier_from_percentile(percentile: float) -> str:
    """Pick a tier label given a player's percentile rank in [0, 100]."""
    for threshold, label in TIER_PERCENTILES:
        if percentile >= threshold:
            return label
    return TIER_PERCENTILES[-1][1]


def apply_percentile_tiers(
    results: list[tuple["PlayerData", float, "ScoreBreakdown"]],
) -> list[tuple["PlayerData", float, "ScoreBreakdown"]]:
    """Rewrite each breakdown.tier in-place using percentile bands.

    Expects `results` sorted descending by score (as returned by
    `FPLScorer.score_squad`). Top-scored player gets the highest percentile.
    """
    n = len(results)
    if n <= 1:
        return results
    for i, (_, _, breakdown) in enumerate(results):
        percentile = 100 * (n - 1 - i) / (n - 1)
        breakdown.tier = tier_from_percentile(percentile)
    return results


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixtureInfo:
    difficulty: int          # FDR 1–5
    is_home: bool
    gameweek: int


@dataclass
class PlayerData:
    """
    Mirrors the FPL bootstrap-static player object.
    All fields map directly to the FPL API response.
    """
    id: int
    web_name: str
    element_type: int            # 1=GK, 2=DEF, 3=MID, 4=FWD
    now_cost: int                # in tenths (e.g. 55 = £5.5m)
    total_points: int
    form: str                    # rolling avg as string e.g. "7.2"
    selected_by_percent: str     # ownership % as string e.g. "12.4"
    transfers_in_event: int
    transfers_out_event: int
    chance_of_playing_next_round: Optional[int]  # 0,25,50,75,100 or None
    bonus: int                   # total bonus points this season
    minutes: int                 # total minutes played

    # Optional enriched stats (from Understat/FBRef)
    xg: float = 0.0
    xa: float = 0.0

    # Clean sheet stats
    clean_sheets: int = 0
    clean_sheet_opportunities: int = 0  # matches played (non-zero min)

    # Set-piece duties — FPL exposes these as taker order
    # (1 = primary, 2 = secondary, 3 = tertiary, None = not on duties)
    penalties_order:                       Optional[int] = None
    direct_freekicks_order:                Optional[int] = None
    corners_and_indirect_freekicks_order:  Optional[int] = None


_ORDER_LABEL = {1: "primary", 2: "secondary", 3: "tertiary"}


def derive_set_piece_role(
    pen_order: Optional[int],
    fk_order: Optional[int],
    corner_order: Optional[int],
) -> str:
    """Human-readable summary of a player's set-piece duties (e.g. for prompts)."""
    parts: list[str] = []
    if pen_order in _ORDER_LABEL:
        parts.append(f"{_ORDER_LABEL[pen_order]} pen")
    if fk_order in _ORDER_LABEL:
        parts.append(f"{_ORDER_LABEL[fk_order]} FK")
    if corner_order in _ORDER_LABEL:
        parts.append(f"{_ORDER_LABEL[corner_order]} corners")
    return ", ".join(parts) if parts else "none"


@dataclass
class ScoreBreakdown:
    fixture:      float = 0.0
    form:         float = 0.0
    value:        float = 0.0
    home_away:    float = 0.0
    xg_xa:        float = 0.0
    availability: float = 0.0
    momentum:     float = 0.0
    clean_sheet:  float = 0.0
    bonus_rate:   float = 0.0
    set_piece:    float = 0.0
    sentiment:    float = 0.0   # external input, not weighted by default
    total:        float = 0.0
    tier:         str = ""
    position:     str = ""
    weights_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual Factor Calculators
# ---------------------------------------------------------------------------

class FactorCalculators:

    @staticmethod
    def fixture_score(fixtures: list[FixtureInfo], n: int = 5) -> float:
        """
        Scores upcoming fixture difficulty using decay weighting.
        Lower FDR = easier = higher score.
        FDR scale: 1 (very easy) → 5 (very hard)
        Returns: 0.0 (hardest) → 1.0 (easiest)
        """
        if not fixtures:
            return 0.5  # neutral if no fixtures (blank GW etc.)

        n = min(n, len(fixtures), len(FIXTURE_DECAY))
        decay = FIXTURE_DECAY[:n]
        total_weight = sum(decay)

        weighted_fdr = sum(
            fixtures[i].difficulty * decay[i]
            for i in range(n)
        ) / total_weight

        # FDR 1→1.0, FDR 5→0.0
        return round(1 - (weighted_fdr - 1) / 4, 4)

    @staticmethod
    def form_score(form_str: str, max_form: float = 12.0) -> float:
        """
        Normalises FPL form (rolling 30-day avg points) to 0–1.
        A form of 12+ is capped at 1.0.
        """
        try:
            form = float(form_str)
        except (ValueError, TypeError):
            return 0.0
        return round(min(form / max_form, 1.0), 4)

    @staticmethod
    def value_score(total_points: int, price_millions: float) -> float:
        """
        Points per million — rewards budget differential picks.
        Normalised against a benchmark of ~20 pts/£m for top performers.
        """
        if price_millions <= 0:
            return 0.0
        ppm = total_points / price_millions
        return round(min(ppm / 20.0, 1.0), 4)

    @staticmethod
    def home_away_score(fixtures: list[FixtureInfo], n: int = 5) -> float:
        """
        Proportion of upcoming n fixtures that are at home.
        All home = 1.0, all away = 0.0.
        Applies a small bonus curve — being mostly home is good but not linear.
        """
        if not fixtures:
            return 0.5

        n = min(n, len(fixtures))
        home_count = sum(1 for f in fixtures[:n] if f.is_home)
        raw = home_count / n

        # Slight S-curve to reward heavy home runs more
        return round(1 / (1 + math.exp(-6 * (raw - 0.5))), 4)

    @staticmethod
    def xg_xa_score(
        xg: float,
        xa: float,
        position: Position,
        minutes: int,
        max_xg_per90: float = 0.6,
        max_xa_per90: float = 0.5,
    ) -> float:
        """
        Expected goal contribution normalised per 90 minutes.
        Weights xG vs xA differently per position.
        """
        if minutes < 90:
            return 0.0

        per90 = minutes / 90
        xg_per90 = xg / per90
        xa_per90 = xa / per90

        # Position-specific xG/xA blend
        blends = {
            Position.GOALKEEPER:  (0.0, 0.0),
            Position.DEFENDER:    (0.3, 0.7),
            Position.MIDFIELDER:  (0.5, 0.5),
            Position.FORWARD:     (0.7, 0.3),
        }
        xg_w, xa_w = blends.get(position, (0.5, 0.5))

        xg_norm = min(xg_per90 / max_xg_per90, 1.0)
        xa_norm = min(xa_per90 / max_xa_per90, 1.0)

        return round(xg_w * xg_norm + xa_w * xa_norm, 4)

    @staticmethod
    def availability_score(chance_of_playing: Optional[int]) -> float:
        """
        Converts FPL's chance-of-playing field to a 0–1 score.
        None means no injury concern = 100%.
        Penalises doubts harshly (25% chance = 0.1 score, not 0.25).
        """
        if chance_of_playing is None:
            return 1.0

        # Apply exponential penalty for injury concerns
        raw = chance_of_playing / 100
        return round(raw ** 1.5, 4)  # 100%→1.0, 75%→0.65, 50%→0.35, 25%→0.12

    @staticmethod
    def momentum_score(
        transfers_in: int,
        transfers_out: int,
        selected_pct: str,
        max_net: int = 300_000,
        max_ownership: float = 60.0,
    ) -> float:
        """
        Combines net transfer activity and ownership into a momentum signal.
        Net positive transfers = buying pressure = community confidence.
        Ownership provides context (low ownership + high buy-ins = differential gem).
        """
        try:
            ownership = float(selected_pct)
        except (ValueError, TypeError):
            ownership = 0.0

        net = transfers_in - transfers_out
        net_norm = max(min(net / max_net, 1.0), -1.0)       # -1 to 1
        ownership_norm = min(ownership / max_ownership, 1.0) # 0 to 1

        # Shift net from [-1,1] to [0,1]
        net_shifted = (net_norm + 1) / 2

        return round(0.65 * net_shifted + 0.35 * ownership_norm, 4)

    @staticmethod
    def clean_sheet_score(clean_sheets: int, opportunities: int) -> float:
        """
        Clean sheet rate over the season with a Laplace smoothing
        to avoid extremes for small samples.
        """
        if opportunities <= 0:
            return 0.0
        # Laplace smoothing: add 1 to numerator and denominator
        rate = (clean_sheets + 1) / (opportunities + 2)
        return round(rate, 4)

    @staticmethod
    def bonus_rate_score(bonus: int, minutes: int) -> float:
        """
        Bonus points per 90 minutes played — rewards consistent
        high-performing players who attract bonus.
        Normalised against a benchmark of 1.0 bonus pts per 90.
        """
        if minutes < 90:
            return 0.0
        per90 = bonus / (minutes / 90)
        return round(min(per90 / 1.0, 1.0), 4)

    # Set-piece taker weights — penalties carry by far the biggest expected value
    # (~75% conversion, certain shots), direct FKs next, corners last (assist-only).
    _SET_PIECE_ROLE_WEIGHT = {
        "penalties": 1.00,
        "direct_freekicks": 0.60,
        "corners": 0.40,
    }
    # Order discount: secondary takers get a chance only when primary is absent
    # or yields the duty; tertiary is largely cosmetic.
    _SET_PIECE_ORDER_DECAY = {1: 1.00, 2: 0.50, 3: 0.25}

    @classmethod
    def set_piece_score(
        cls,
        penalties_order: Optional[int],
        direct_freekicks_order: Optional[int],
        corners_order: Optional[int],
    ) -> float:
        """
        Score the player's set-piece duties on a 0–1 scale.

        Takes the *max* contribution across the three roles — a primary penalty
        taker scores 1.0 regardless of corners, and we don't double-count someone
        who happens to be on multiple set-piece duties.
        """
        contributions: list[float] = []
        for order, role in (
            (penalties_order,        "penalties"),
            (direct_freekicks_order, "direct_freekicks"),
            (corners_order,          "corners"),
        ):
            decay = cls._SET_PIECE_ORDER_DECAY.get(order)
            if decay is None:
                continue
            contributions.append(cls._SET_PIECE_ROLE_WEIGHT[role] * decay)
        if not contributions:
            return 0.0
        return round(min(max(contributions), 1.0), 4)


# ---------------------------------------------------------------------------
# Main Scorer
# ---------------------------------------------------------------------------

class FPLScorer:
    """
    Main scoring engine. Computes a composite score for a player
    based on multiple weighted factors.

    Example:
        scorer = FPLScorer()
        total, breakdown = scorer.score_player(player, fixtures, sentiment=0.7)
        print(f"{player.web_name}: {total:.3f} — {breakdown.tier}")
    """

    def __init__(self, custom_weights: Optional[dict] = None):
        """
        Args:
            custom_weights: Override default position weights with your own.
                            Must be a dict matching POSITION_WEIGHTS structure.
        """
        self.custom_weights = custom_weights
        self.calc = FactorCalculators()

    def _get_weights(self, position: Position) -> dict:
        if self.custom_weights:
            return self.custom_weights.get(position, BASE_WEIGHTS)
        return POSITION_WEIGHTS.get(position, BASE_WEIGHTS)

    def _get_position(self, element_type: int) -> Position:
        return Position(element_type)

    def _get_tier(self, score: float) -> str:
        for threshold, label in TIERS:
            if score >= threshold:
                return label
        return TIERS[-1][1]

    def score_player(
        self,
        player: PlayerData,
        fixtures: list[FixtureInfo],
        sentiment: float = 0.5,
        sentiment_weight: float = 0.05,
        n_fixtures: int = 5,
    ) -> tuple[float, ScoreBreakdown]:
        """
        Compute the composite score for a single player.

        Args:
            player:           PlayerData object from FPL API
            fixtures:         List of upcoming FixtureInfo objects
            sentiment:        External sentiment score 0.0–1.0 (from Hermes/Reddit)
            sentiment_weight: How much to weight sentiment in final score (default 5%)
            n_fixtures:       How many upcoming fixtures to evaluate (default 5)

        Returns:
            (total_score: float, breakdown: ScoreBreakdown)
        """
        position = self._get_position(player.element_type)
        weights = self._get_weights(position)
        price = player.now_cost / 10  # convert to millions

        # --- Calculate each factor ---
        factors = {
            "fixture":      self.calc.fixture_score(fixtures, n_fixtures),
            "form":         self.calc.form_score(player.form),
            "value":        self.calc.value_score(player.total_points, price),
            "home_away":    self.calc.home_away_score(fixtures, n_fixtures),
            "xg_xa":        self.calc.xg_xa_score(player.xg, player.xa, position, player.minutes),
            "availability": self.calc.availability_score(player.chance_of_playing_next_round),
            "momentum":     self.calc.momentum_score(
                                player.transfers_in_event,
                                player.transfers_out_event,
                                player.selected_by_percent,
                            ),
            "clean_sheet":  self.calc.clean_sheet_score(
                                player.clean_sheets,
                                player.clean_sheet_opportunities,
                            ),
            "bonus_rate":   self.calc.bonus_rate_score(player.bonus, player.minutes),
            "set_piece":    self.calc.set_piece_score(
                                player.penalties_order,
                                player.direct_freekicks_order,
                                player.corners_and_indirect_freekicks_order,
                            ),
        }

        # --- Weighted sum ---
        base_score = sum(factors[k] * weights[k] for k in factors)

        # --- Blend in sentiment ---
        total = (base_score * (1 - sentiment_weight)) + (sentiment * sentiment_weight)
        total = round(min(max(total, 0.0), 1.0), 4)

        breakdown = ScoreBreakdown(
            fixture=      factors["fixture"],
            form=         factors["form"],
            value=        factors["value"],
            home_away=    factors["home_away"],
            xg_xa=        factors["xg_xa"],
            availability= factors["availability"],
            momentum=     factors["momentum"],
            clean_sheet=  factors["clean_sheet"],
            bonus_rate=   factors["bonus_rate"],
            set_piece=    factors["set_piece"],
            sentiment=    round(sentiment, 4),
            total=        total,
            tier=         self._get_tier(total),
            position=     position.name,
            weights_used= weights,
        )

        return total, breakdown

    def score_squad(
        self,
        players: list[PlayerData],
        fixtures_map: dict[int, list[FixtureInfo]],
        sentiment_map: dict[int, float] = None,
        **kwargs,
    ) -> list[tuple[PlayerData, float, ScoreBreakdown]]:
        """
        Score an entire squad or player pool, sorted by score descending.

        Args:
            players:       List of PlayerData objects
            fixtures_map:  Dict mapping player_id → list of FixtureInfo
            sentiment_map: Dict mapping player_id → sentiment float (optional)
            **kwargs:      Passed through to score_player()

        Returns:
            Sorted list of (player, score, breakdown) tuples
        """
        if sentiment_map is None:
            sentiment_map = {}

        results = []
        for player in players:
            fixtures = fixtures_map.get(player.id, [])
            sentiment = sentiment_map.get(player.id, 0.5)
            score, breakdown = self.score_player(player, fixtures, sentiment, **kwargs)
            results.append((player, score, breakdown))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def compare_players(
        self,
        player_a: PlayerData,
        player_b: PlayerData,
        fixtures_a: list[FixtureInfo],
        fixtures_b: list[FixtureInfo],
        sentiment_a: float = 0.5,
        sentiment_b: float = 0.5,
    ) -> dict:
        """
        Head-to-head comparison between two players for transfer decisions.

        Returns a dict with scores, breakdowns, winner, and margin.
        """
        score_a, bd_a = self.score_player(player_a, fixtures_a, sentiment_a)
        score_b, bd_b = self.score_player(player_b, fixtures_b, sentiment_b)

        winner = player_a if score_a >= score_b else player_b
        margin = round(abs(score_a - score_b), 4)

        return {
            "player_a": {"player": player_a, "score": score_a, "breakdown": bd_a},
            "player_b": {"player": player_b, "score": score_b, "breakdown": bd_b},
            "winner":   winner.web_name,
            "margin":   margin,
            "confidence": "HIGH" if margin > 0.1 else "MEDIUM" if margin > 0.05 else "LOW",
        }

    def generate_recommendation(
        self,
        player: PlayerData,
        breakdown: ScoreBreakdown,
        score: float,
    ) -> str:
        """
        Generate a human-readable recommendation string for the Hermes agent
        to use as context when formulating its response.
        """
        lines = [
            f"Player: {player.web_name} ({breakdown.position})",
            f"Price: £{player.now_cost / 10:.1f}m",
            f"Score: {score:.3f} — {breakdown.tier}",
            "",
            "Factor Breakdown:",
            f"  Fixtures:       {breakdown.fixture:.2f}  (weight: {breakdown.weights_used.get('fixture', 0):.0%})",
            f"  Form:           {breakdown.form:.2f}  (weight: {breakdown.weights_used.get('form', 0):.0%})",
            f"  Value (PPM):    {breakdown.value:.2f}  (weight: {breakdown.weights_used.get('value', 0):.0%})",
            f"  Home/Away:      {breakdown.home_away:.2f}  (weight: {breakdown.weights_used.get('home_away', 0):.0%})",
            f"  xG/xA:         {breakdown.xg_xa:.2f}  (weight: {breakdown.weights_used.get('xg_xa', 0):.0%})",
            f"  Availability:   {breakdown.availability:.2f}  (weight: {breakdown.weights_used.get('availability', 0):.0%})",
            f"  Momentum:       {breakdown.momentum:.2f}  (weight: {breakdown.weights_used.get('momentum', 0):.0%})",
            f"  Clean Sheets:   {breakdown.clean_sheet:.2f}  (weight: {breakdown.weights_used.get('clean_sheet', 0):.0%})",
            f"  Bonus Rate:     {breakdown.bonus_rate:.2f}  (weight: {breakdown.weights_used.get('bonus_rate', 0):.0%})",
            f"  Set Pieces:     {breakdown.set_piece:.2f}  (weight: {breakdown.weights_used.get('set_piece', 0):.0%})",
            f"  Sentiment:      {breakdown.sentiment:.2f}  (community signal)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FPL API Helper — maps raw API data to typed objects
# ---------------------------------------------------------------------------

class FPLDataMapper:
    """
    Converts raw FPL API JSON responses into typed scoring module objects.
    """

    @staticmethod
    def player_from_api(data: dict) -> PlayerData:
        """Map a player element from bootstrap-static."""
        def _to_float(value, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return PlayerData(
            id=                             data["id"],
            web_name=                       data["web_name"],
            element_type=                   data["element_type"],
            now_cost=                       data["now_cost"],
            total_points=                   data["total_points"],
            form=                           data.get("form", "0"),
            selected_by_percent=            data.get("selected_by_percent", "0"),
            transfers_in_event=             data.get("transfers_in_event", 0),
            transfers_out_event=            data.get("transfers_out_event", 0),
            chance_of_playing_next_round=   data.get("chance_of_playing_next_round"),
            bonus=                          data.get("bonus", 0),
            minutes=                        data.get("minutes", 0),
            # FPL exposes expected stats as strings (e.g. "12.43") on bootstrap elements
            xg=                             _to_float(data.get("expected_goals")),
            xa=                             _to_float(data.get("expected_assists")),
            clean_sheets=                   data.get("clean_sheets", 0),
            # clean_sheet_opportunities approximated by games started
            clean_sheet_opportunities=      data.get("starts", 0),
            # Set-piece taker order (1=primary, 2=secondary, 3=tertiary, None=off duty)
            penalties_order=                      data.get("penalties_order"),
            direct_freekicks_order=               data.get("direct_freekicks_order"),
            corners_and_indirect_freekicks_order= data.get(
                "corners_and_indirect_freekicks_order"
            ),
        )

    @staticmethod
    def fixtures_from_api(
        fixture_list: list[dict],
        team_id: int,
        n: int = 5,
    ) -> list[FixtureInfo]:
        """
        Extract upcoming fixtures for a team from the fixtures endpoint.
        fixture_list: raw list from /api/fixtures/?event={gw}
        """
        upcoming = []
        for f in fixture_list:
            if f.get("finished"):
                continue
            if f["team_h"] == team_id:
                upcoming.append(FixtureInfo(
                    difficulty= f["team_h_difficulty"],
                    is_home=    True,
                    gameweek=   f.get("event", 0),
                ))
            elif f["team_a"] == team_id:
                upcoming.append(FixtureInfo(
                    difficulty= f["team_a_difficulty"],
                    is_home=    False,
                    gameweek=   f.get("event", 0),
                ))
            if len(upcoming) >= n:
                break
        return upcoming


# ---------------------------------------------------------------------------
# Quick demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Example player (Salah-like stats)
    salah = PlayerData(
        id=233,
        web_name="Salah",
        element_type=3,         # MID
        now_cost=130,           # £13.0m
        total_points=180,
        form="9.2",
        selected_by_percent="48.5",
        transfers_in_event=180000,
        transfers_out_event=25000,
        chance_of_playing_next_round=None,
        bonus=28,
        minutes=2430,
        xg=12.4,
        xa=8.1,
        clean_sheets=4,
        clean_sheet_opportunities=27,
        penalties_order=1,
        direct_freekicks_order=1,
        corners_and_indirect_freekicks_order=2,
    )

    # Easy upcoming fixture run
    fixtures = [
        FixtureInfo(difficulty=2, is_home=True,  gameweek=30),
        FixtureInfo(difficulty=2, is_home=False, gameweek=31),
        FixtureInfo(difficulty=3, is_home=True,  gameweek=32),
        FixtureInfo(difficulty=2, is_home=True,  gameweek=33),
        FixtureInfo(difficulty=4, is_home=False, gameweek=34),
    ]

    scorer = FPLScorer()
    score, breakdown = scorer.score_player(salah, fixtures, sentiment=0.78)

    print(scorer.generate_recommendation(salah, breakdown, score))
    print(f"\nOverall Score: {score:.4f}")