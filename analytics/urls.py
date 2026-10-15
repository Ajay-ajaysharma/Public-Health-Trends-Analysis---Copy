from django.urls import path
from . import views

urlpatterns = [
    path('', views.overview_view, name='overview'),
    path('trends/', views.trends_view, name='trends'),
    path('country-explorer/', views.country_explorer_view, name='country_explorer'),
    path('risk-map/', views.risk_map_view, name='risk_map'),
    path('simulator/', views.simulator_view, name='simulator'),
    path('model-evaluation/', views.model_evaluation_view, name='model_evaluation'),
    
    # REST API Endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/country-data/<str:country_name>/', views.api_country_data, name='api_country_data'),
]
