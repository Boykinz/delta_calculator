class LLMReportEnhancer:
    """Улучшение отчёта с помощью LLM"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_executive_summary(self, results: Dict) -> str:
        """Генерация executive summary для руководства"""
        prompt = f"""
Ты - финансовый аналитик. На основе следующих данных подготовь краткое резюме для руководства компании.

Данные:
- Базовый NPV проекта: {results['baseline']['npv']:,.0f} руб.
- NPV с мерами поддержки: {results['cumulative']['npv']:,.0f} руб.
- Прирост NPV: {results['delta']['npv']:,.0f} руб.
- Сокращение срока окупаемости: {results['delta']['payback']:.1f} лет
- Количество мер: {results['total_measures']}

Требования:
1. Краткость (3-4 абзаца)
2. Акцент на ключевых выводах
3. Рекомендации по приоритетным мерам
4. Указание на риски (если есть)
"""
        response = self.llm.generate(prompt)
        return response
    
    def generate_risk_analysis(self, results: Dict) -> str:
        """Анализ рисков и ограничений"""
        prompt = f"""
Проанализируй следующие меры поддержки и укажи потенциальные риски:

Меры: {[e.measure_name for e in results['individual_effects']]}

Укажи:
1. Риски неполучения мер (условия, сроки)
2. Риски изменения законодательства
3. Ограничения кумулятивного применения
"""
        response = self.llm.generate(prompt)
        return response
