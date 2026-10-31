from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from . import views

urlpatterns = [
    path("prediccion/", views.PrediccionAPIView.as_view(), name="api_prediccion"),
    path("probabilidades/", views.ProbabilidadesAPIView.as_view(), name="api_probabilidades"),
    path("schema/", SpectacularAPIView.as_view(), name="api_schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="api_schema"), name="api_docs"),
]
