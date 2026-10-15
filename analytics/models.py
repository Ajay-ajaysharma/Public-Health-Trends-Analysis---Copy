from django.db import models

class SimulationLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    country_name = models.CharField(max_length=120, blank=True, null=True)
    who_region = models.CharField(max_length=80)
    income_group = models.CharField(max_length=80)
    dtp3_coverage = models.FloatField()
    measles_coverage = models.FloatField()
    physicians_per_10k = models.FloatField()
    nurses_per_10k = models.FloatField()
    hospital_beds_per_10k = models.FloatField()
    pm25_pollution = models.FloatField()
    clean_water_pct = models.FloatField()
    gdp_per_capita = models.FloatField()
    predicted_high_risk = models.BooleanField()
    high_risk_probability = models.FloatField()

    def __str__(self):
        return f"{self.country_name or 'Custom'} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} (Risk: {self.high_risk_probability:.2f})"
