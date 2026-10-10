from decimal import Decimal

import pytest

from evals.s2.revision_budget import CampaignBudget, EvalBudgetLimit


def test_budget_reserves_before_the_next_provider_call():
    budget = CampaignBudget(Decimal("0.50"))
    budget.record_known_charge(Decimal("0.44"))
    with pytest.raises(EvalBudgetLimit):
        budget.reserve(Decimal("0.02"))
    assert budget.liability_usd == Decimal("0.44")


def test_unknown_usage_retains_reserved_cost_and_stops_calls():
    budget = CampaignBudget(Decimal("0.50"))
    budget.reserve(Decimal("0.10"))
    budget.unknown_charge()
    with pytest.raises(EvalBudgetLimit):
        budget.reserve(Decimal("0.01"))
    assert budget.liability_usd == Decimal("0.10")


def test_resume_raises_cumulative_limit_without_resetting_spend(tmp_path):
    path = tmp_path / "campaign.json"
    first = CampaignBudget(Decimal("0.50"), path=path)
    first.record_known_charge(Decimal("0.30"))
    key = first.reserve(Decimal("0.10"))
    first.reconcile(key, Decimal("0.04"))
    resumed = CampaignBudget(Decimal("1.00"), path=path)
    assert resumed.liability_usd == Decimal("0.34")
    assert resumed.limit == Decimal("0.90")
