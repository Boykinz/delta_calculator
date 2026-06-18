import numpy as np
from scipy.optimize import brentq
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import warnings

# ==============================================================================
# 1. МОДЕЛИ ДАННЫХ (Словарь системы)
# ==============================================================================

class StatementType(Enum):
    PNL = "P&L"
    BS = "Balance Sheet"
    CFS = "Cash Flow Statement"

class FinancialProjections(BaseModel):
    """Ключевые финансовые прогнозы (упрощенная структура для 3-х форм)"""
    horizon: int = Field(..., description="Горизонт планирования (лет)")
    step: float = Field(1.0, description="Шаг планирования (доля года)")
    
    # P&L (Отчет о прибылях и убытках)
    revenue: List[float]
    opex: List[float] # Включая налоги на имущество/землю, взносы
    depreciation: List[float]
    interest_expense: List[float]
    profit_tax_rate: float
    property_tax_rate: float
    
    # CFS (Отчет о движении денежных средств)
    capex: List[float]
    working_capital_change: List[float]
    
    # BS (Баланс) - для проверки сходимости
    initial_equity: float
    initial_debt: float
    initial_fixed_assets: float

class DebtSchedule(BaseModel):
    """Условия финансирования и график долга"""
    tranches: List[Dict[str, Any]] # e.g., {"amount": 100, "rate": 0.10, "grace_period": 2, "tenor": 10}
    # В реальном движке здесь будет полноценный расчет аннуитета/дифференцированных платежей

class DiscountParameters(BaseModel):
    """Параметры дисконтирования"""
    wacc: float = Field(..., description="Ставка дисконтирования проекта (WACC)")
    ke: float = Field(..., description="Стоимость собственного капитала (Ke)")
    risk_free_rate: float = 0.0
    inflation_rate: float = 0.0

class ImpactRule(BaseModel):
    """Правило влияния МП на финансовую модель (п. 7.3)"""
    target: str # e.g., "opex", "capex", "debt_rate", "profit_tax"
    line_item: Optional[str] = None # e.g., "property_tax", "equipment"
    operation: str # "multiply", "subtract", "replace", "add_inflow"
    value: float
    cost_item_id: str # Уникальный ID статьи затрат для контроля двойного финансирования

class SupportMeasureCard(BaseModel):
    """Карточка Меры Поддержки (МП)"""
    id: str
    name: str
    npa_source: str
    measure_type: str # loan, subsidy, tax_break, guarantee, etc.
    impact_rules: List[ImpactRule]
    priority: int = 1 # Приоритет для последовательной атрибуции

# ==============================================================================
# 2. КОНТРОЛЬ КАЧЕСТВА И СХОДИМОСТИ (п. 5.4.3.2 и п. 7.5)
# ==============================================================================

class ConsistencyChecker:
    """Проверка сходимости трех форм отчетности и базовых тождеств"""
    
    @staticmethod
    def check_3_statement_balance(proj: FinancialProjections, debt_schedule: DebtSchedule) -> List[str]:
        errors = []
        # Упрощенная проверка: Чистая прибыль должна корректно переходить в накопленный капитал
        # и изменение долга в CFS должно совпадать с изменением долга в BS.
        
        # 1. Проверка неотрицательности ставок
        if proj.profit_tax_rate < 0 or proj.property_tax_rate < 0:
            errors.append("CRITICAL: Отрицательные налоговые ставки.")
            
        # 2. Проверка логики CAPEX и Амортизации
        total_capex = sum(proj.capex)
        total_dep = sum(proj.depreciation)
        if total_dep > total_capex + proj.initial_fixed_assets:
            errors.append("WARNING: Суммарная амортизация превышает CAPEX + начальные ОС.")
            
        return errors

class QualityControl:
    """Численные и бизнес-проверки результатов (п. 7.5)"""
    
    @staticmethod
    def check_irr_uniqueness(cash_flows: List[float]) -> Tuple[bool, str]:
        """Проверка нестандартных потоков (множественная смена знака)"""
        signs = [np.sign(cf) for cf in cash_flows if cf != 0]
        sign_changes = sum(1 for i in range(len(signs)-1) if signs[i] != signs[i+1])
        
        if sign_changes > 1:
            return False, "ВНИМАНИЕ: Множественная смена знака в ДП. IRR может быть неоднозначен. Используйте NPV."
        return True, "OK"
        
    @staticmethod
    def check_npv_irr_consistency(npv: float, irr: float, discount_rate: float) -> str:
        """Согласованность NPV и IRR"""
        if irr is None: return "N/A"
        if irr > discount_rate and npv < 0:
            return "WARNING: IRR > WACC, но NPV < 0 (возможна ошибка в расчетах или нестандартный поток)."
        if irr < discount_rate and npv > 0:
            return "WARNING: IRR < WACC, но NPV > 0."
        return "OK"

# ==============================================================================
# 3. ФИНАНСОВЫЙ ДВИЖОК (Ядро расчетов)
# ==============================================================================

class FinancialEngine:
    """Движок расчета базовой экономики и метрик (Модуль 4)"""
    
    def __init__(self, projections: FinancialProjections, debt: DebtSchedule, discount: DiscountParameters):
        self.proj = projections
        self.debt = debt
        self.discount = discount
        self.assumptions_registry = [] # Реестр допущений (п. 5.4.3.8)
        self.warnings = []
        
    def _calculate_ebit_and_tax(self, opex_mod: List[float], tax_rate_mod: float) -> Tuple[List[float], List[float]]:
        ebit = [self.proj.revenue[i] - opex_mod[i] - self.proj.depreciation[i] for i in range(self.proj.horizon)]
        tax = [max(0, ebit[i] * tax_rate_mod) for i in range(self.proj.horizon)]
        return ebit, tax

    def calculate_fcff(self, opex_mod: List[float], capex_mod: List[float], tax_rate_mod: float) -> List[float]:
        """Расчет FCFF (Свободный денежный поток фирмы)"""
        ebit, tax = self._calculate_ebit_and_tax(opex_mod, tax_rate_mod)
        fcff = []
        for i in range(self.proj.horizon):
            # FCFF = EBIT * (1 - t) + D&A - CAPEX - delta NWC
            nopat = ebit[i] * (1 - tax_rate_mod)
            cf = nopat + self.proj.depreciation[i] - capex_mod[i] - self.proj.working_capital_change[i]
            fcff.append(cf)
        return fcff

    def calculate_fcfe(self, fcff: List[float], interest_mod: List[float], tax_rate_mod: float) -> List[float]:
        """Расчет FCFE (Свободный денежный поток на собственный капитал)"""
        # В упрощенной модели: FCFE = FCFF - Int*(1-t) + Net Borrowing
        # Для точной модели нужен полный Debt Schedule. Здесь используем базовый подход.
        net_borrowing = [0] * self.proj.horizon # Заглушка, в реальности берется из графика долга
        net_borrowing[0] = self.proj.initial_debt # Получение кредита в 0 год
        
        fcfe = []
        for i in range(self.proj.horizon):
            cf = fcff[i] - interest_mod[i] * (1 - tax_rate_mod) + net_borrowing[i]
            fcfe.append(cf)
        return fcfe

    @staticmethod
    def _calc_npv(cfs: List[float], rate: float) -> float:
        return sum(cf / (1 + rate)**t for t, cf in enumerate(cfs))

    @staticmethod
    def _calc_irr(cfs: List[float]) -> Optional[float]:
        try:
            # Используем brentq для поиска корня NPV(r) = 0
            def npv_func(r): return sum(cf / (1 + r)**t for t, cf in enumerate(cfs))
            # Ищем в диапазоне от -50% до +100%
            return brentq(npv_func, -0.5, 1.0)
        except ValueError:
            return None # IRR не найден

    @staticmethod
    def _calc_dpp(cfs: List[float], rate: float) -> Optional[float]:
        """Дисконтированный срок окупаемости (DPP)"""
        cum_disc_cf = 0
        for t, cf in enumerate(cfs):
            disc_cf = cf / (1 + rate)**t
            prev_cum = cum_disc_cf
            cum_disc_cf += disc_cf
            if cum_disc_cf >= 0 and prev_cum < 0:
                # Линейная интерполяция
                return t - 1 + (-prev_cum / disc_cf)
        return None # Не окупается

    def calculate_metrics(self, cfs: List[float], rate: float, label: str) -> Dict[str, Any]:
        """Расчет NPV, IRR, DPP с контролем качества"""
        npv = self._calc_npv(cfs, rate)
        irr = self._calc_irr(cfs)
        dpp = self._calc_dpp(cfs, rate)
        
        # Контроль качества (п. 7.5)
        is_unique, irr_warn = QualityControl.check_irr_uniqueness(cfs)
        if not is_unique: self.warnings.append(f"{label}: {irr_warn}")
            
        consistency = QualityControl.check_npv_irr_consistency(npv, irr, rate)
        if "WARNING" in consistency: self.warnings.append(f"{label}: {consistency}")

        return {"NPV": npv, "IRR": irr, "DPP": dpp, "CFs": cfs}

# ==============================================================================
# 4. ДВИЖОК ПРИМЕНЕНИЯ МП И КОНТРОЛЬ ДВОЙНОГО ФИНАНСИРОВАНИЯ (п. 7.3, 7.4)
# ==============================================================================

class CostSubsidyTracker:
    """Трекер для запрета двойного финансирования (п. 7.4.1)"""
    def __init__(self):
        self.subsidized_items = {} # {cost_item_id: [measure_id1, measure_id2]}

    def check_and_register(self, cost_item_id: str, measure_id: str) -> bool:
        if cost_item_id in self.subsidized_items:
            return False # Двойное финансирование!
        self.subsidized_items[cost_item_id] = measure_id
        return True

class MeasureApplicator:
    """Применение эффектов МП к финансовой модели (п. 7.3)"""
    
    def __init__(self, base_projections: FinancialProjections):
        self.base = base_projections
        self.mod_opex = base_projections.opex.copy()
        self.mod_capex = base_projections.capex.copy()
        self.mod_interest = base_projections.interest_expense.copy()
        self.mod_tax_rate = base_projections.profit_tax_rate
        self.tracker = CostSubsidyTracker()
        self.applied_adjustments = [] # Для аудируемости

    def apply_measure(self, card: SupportMeasureCard) -> bool:
        """Применяет правила из Карточки МП. Возвращает False если заблокировано трекером."""
        for rule in card.impact_rules:
            # 1. Проверка на двойное финансирование
            if rule.operation in ["subtract", "multiply"] and rule.target in ["opex", "capex"]:
                if not self.tracker.check_and_register(rule.cost_item_id, card.id):
                    self.applied_adjustments.append(f"BLOCKED: {card.name} -> {rule.cost_item_id} (Двойное финансирование)")
                    return False
            
            # 2. Применение правила (Динамическая параметризация)
            if rule.target == "opex" and rule.line_item == "property_tax":
                if rule.operation == "multiply":
                    reduction = sum(self.mod_opex) * (1 - rule.value) # Упрощенно
                    self.mod_opex = [x * rule.value for x in self.mod_opex]
                    self.applied_adjustments.append(f"{card.name}: OPEX (налог на имущество) * {rule.value}")
                    
            elif rule.target == "profit_tax":
                if rule.operation == "replace":
                    self.mod_tax_rate = rule.value
                    self.applied_adjustments.append(f"{card.name}: Ставка налога на прибыль = {rule.value}")
                    
            elif rule.target == "debt_rate":
                if rule.operation == "replace":
                    self.mod_interest = [x * (rule.value / 0.10) for x in self.mod_interest] # Пропорционально
                    self.applied_adjustments.append(f"{card.name}: Ставка по кредиту снижена до {rule.value}")
                    
            # Здесь добавляются остальные правила из п. 7.3 (инвест. вычет, гранты и т.д.)
            
        return True

# ==============================================================================
# 5. ДВИЖОК АТРИБУЦИИ (ДЕКОМПОЗИЦИИ) ЭФФЕКТА (п. 7.4)
# ==============================================================================

class AttributionEngine:
    """Расчет вклада каждой меры в общий эффект (Sequential Method)"""
    
    def __init__(self, engine: FinancialEngine):
        self.engine = engine
        
    def calculate_decomposition(self, measures: List[SupportMeasureCard]) -> Dict[str, Any]:
        # Сортировка по приоритету (п. 7.4.2)
        sorted_measures = sorted(measures, key=lambda m: m.priority)
        
        # 1. Базовый сценарий
        base_applicator = MeasureApplicator(self.engine.proj)
        base_fcff = self.engine.calculate_fcff(base_applicator.mod_opex, base_applicator.mod_capex, base_applicator.mod_tax_rate)
        base_fcfe = self.engine.calculate_fcfe(base_fcff, base_applicator.mod_interest, base_applicator.mod_tax_rate)
        
        base_metrics_proj = self.engine.calculate_metrics(base_fcff, self.engine.discount.wacc, "Base Project")
        base_metrics_eq = self.engine.calculate_metrics(base_fcfe, self.engine.discount.ke, "Base Equity")
        
        results = {
            "base_project": base_metrics_proj,
            "base_equity": base_metrics_eq,
            "measures": [],
            "cumulative_project": None,
            "cumulative_equity": None
        }
        
        # 2. Последовательное применение и расчет дельт
        current_applicator = MeasureApplicator(self.engine.proj)
        prev_npv_proj = base_metrics_proj["NPV"]
        prev_npv_eq = base_metrics_eq["NPV"]
        
        for measure in sorted_measures:
            is_applied = current_applicator.apply_measure(measure)
            
            if is_applied:
                # Пересчет с текущей мерой
                fcff = self.engine.calculate_fcff(current_applicator.mod_opex, current_applicator.mod_capex, current_applicator.mod_tax_rate)
                fcfe = self.engine.calculate_fcfe(fcff, current_applicator.mod_interest, current_applicator.mod_tax_rate)
                
                m_proj = self.engine.calculate_metrics(fcff, self.engine.discount.wacc, f"With {measure.name}")
                m_eq = self.engine.calculate_metrics(fcfe, self.engine.discount.ke, f"With {measure.name}")
                
                # Атрибуция (Delta)
                delta_npv_proj = m_proj["NPV"] - prev_npv_proj
                delta_npv_eq = m_eq["NPV"] - prev_npv_eq
                
                results["measures"].append({
                    "name": measure.name,
                    "npa": measure.npa_source,
                    "delta_npv_project": delta_npv_proj,
                    "delta_npv_equity": delta_npv_eq,
                    "adjustments": current_applicator.applied_adjustments[-len(measure.impact_rules):]
                })
                
                prev_npv_proj = m_proj["NPV"]
                prev_npv_eq = m_eq["NPV"]
            else:
                results["measures"].append({
                    "name": measure.name,
                    "status": "Отклонена (Двойное финансирование или конфликт)",
                    "delta_npv_project": 0,
                    "delta_npv_equity": 0
                })

        # 3. Финальный кумулятивный сценарий (проверка согласованности п. 7.4.3)
        results["cumulative_project"] = self.engine.calculate_metrics(
            self.engine.calculate_fcff(current_applicator.mod_opex, current_applicator.mod_capex, current_applicator.mod_tax_rate),
            self.engine.discount.wacc, "Cumulative Project"
        )
        results["cumulative_equity"] = self.engine.calculate_metrics(
            self.engine.calculate_fcfe(
                self.engine.calculate_fcff(current_applicator.mod_opex, current_applicator.mod_capex, current_applicator.mod_tax_rate),
                current_applicator.mod_interest, current_applicator.mod_tax_rate
            ),
            self.engine.discount.ke, "Cumulative Equity"
        )
        
        # Проверка: Сумма атрибутированных вкладов == Совокупный Δ-эффект
        sum_delta_proj = sum(m.get("delta_npv_project", 0) for m in results["measures"])
        total_delta_proj = results["cumulative_project"]["NPV"] - results["base_project"]["NPV"]
        
        if not np.isclose(sum_delta_proj, total_delta_proj, atol=1e-2):
            self.engine.warnings.append(f"CRITICAL: Рассогласование декомпозиции! Сумма дельт ({sum_delta_proj}) != Общий эффект ({total_delta_proj})")

        return results

# ==============================================================================
# 6. ГЕНЕРАТОР ОТЧЕТА И РЕЕСТРА ДОПУЩЕНИЙ (п. 5.4.3.8)
# ==============================================================================

class AuditReporter:
    """Формирование аудируемого отчета"""
    
    @staticmethod
    def generate_report(results: Dict, engine: FinancialEngine) -> str:
        report = ["=" * 60, "ОТЧЕТ ФИНАНСОВОГО ДВИЖКА (МОДУЛЬ 4)", "=" * 60]
        
        # Реестр допущений
        report.append("\n[РЕЕСТР ДОПУЩЕНИЙ И ПАРАМЕТРОВ]")
        report.append(f"Горизонт: {engine.proj.horizon} лет | Шаг: {engine.proj.step}")
        report.append(f"WACC (Проект): {engine.discount.wacc*100:.2f}% | Ke (Акционер): {engine.discount.ke*100:.2f}%")
        report.append(f"Базовая ставка налога на прибыль: {engine.proj.profit_tax_rate*100:.2f}%")
        
        # Предупреждения
        if engine.warnings:
            report.append("\n[ПРЕДУПРЕЖДЕНИЯ СИСТЕМЫ КОНТРОЛЯ КАЧЕСТВА]")
            for w in engine.warnings:
                report.append(f"⚠️ {w}")
                
        # Сравнение сценариев
        report.append("\n[СРАВНЕНИЕ СЦЕНАРИЕВ]")
        bp = results['base_project']
        cp = results['cumulative_project']
        be = results['base_equity']
        ce = results['cumulative_equity']
        
        report.append(f"{'Показатель':<25} | {'База (Проект)':<15} | {'С МП (Проект)':<15} | {'Δ-эффект':<15}")
        report.append("-" * 75)
        report.append(f"{'NPV (FCFF)':<25} | {bp['NPV']:>15,.0f} | {cp['NPV']:>15,.0f} | {cp['NPV']-bp['NPV']:>15,.0f}")
        report.append(f"{'IRR (FCFF)':<25} | {bp['IRR']*100 if bp['IRR'] else 'N/A':>14.2f}% | {cp['IRR']*100 if cp['IRR'] else 'N/A':>14.2f}% | {(cp['IRR']-bp['IRR'])*100 if bp['IRR'] and cp['IRR'] else 'N/A':>14.2f} п.п.")
        report.append(f"{'DPP (лет)':<25} | {bp['DPP'] if bp['DPP'] else '> Horizon':>15} | {cp['DPP'] if cp['DPP'] else '> Horizon':>15} | {bp['DPP']-cp['DPP'] if bp['DPP'] and cp['DPP'] else 'N/A':>15.1f}")
        
        report.append("\n" + "-" * 75)
        report.append(f"{'NPV (FCFE, Акционер)':<25} | {be['NPV']:>15,.0f} | {ce['NPV']:>15,.0f} | {ce['NPV']-be['NPV']:>15,.0f}")
        
        # Декомпозиция
        report.append("\n[ДЕКОМПОЗИЦИЯ ЭФФЕКТА (АТРИБУЦИЯ)]")
        for m in results['measures']:
            report.append(f"\n▶ {m['name']} (НПА: {m.get('npa', 'N/A')})")
            if m.get('status'):
                report.append(f"  Статус: {m['status']}")
            else:
                report.append(f"  Δ NPV Проекта: {m['delta_npv_project']:>15,.0f} руб.")
                report.append(f"  Δ NPV Акционера: {m['delta_npv_equity']:>15,.0f} руб.")
                report.append(f"  Примененные корректировки: {', '.join(m.get('adjustments', []))}")
                
        return "\n".join(report)
