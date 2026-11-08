from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from fases.modulo import BASE_METRICS, DROP_METRICS, OPTIONAL_METRICS, ReglamentoInvalido, validar_valores_neutros
from fases.montecarlo import probabilidades_campeon
from fases.views import _respuesta_json, leer_metricas, predecir_recepcion

from .serializers import PrediccionSerializer


class PrediccionAPIView(APIView):
    """POST /api/prediccion/ con la misma 'recepcion' que la web"""

    @extend_schema(request=PrediccionSerializer, responses={200: OpenApiTypes.OBJECT},
                   description="Predice el resto del torneo con el modelo que usa la app")
    def post(self, request):
        serializador = PrediccionSerializer(data=request.data)
        serializador.is_valid(raise_exception=True)
        datos = serializador.validated_data
        try:
            respuesta = predecir_recepcion(datos)
        except (ReglamentoInvalido, ValueError) as error:
            return Response({"detail": str(error)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(_respuesta_json(respuesta, datos["recepcion"]))


class ProbabilidadesAPIView(APIView):
    """POST /api/probabilidades/: Etapa G (Montecarlo de los 5 campeones mas probables)"""

    @extend_schema(request=PrediccionSerializer, responses={200: OpenApiTypes.OBJECT},
                   description="Simula el torneo y devuelve los 5 campeones mas probables (Etapa G)")
    def post(self, request):
        serializador = PrediccionSerializer(data=request.data)
        serializador.is_valid(raise_exception=True)
        datos = serializador.validated_data
        recepcion = datos["recepcion"]
        try:
            drop = leer_metricas(datos.get("drop_metricas"), BASE_METRICS, "drop_metricas", DROP_METRICS)
            opcionales = leer_metricas(datos.get("metricas_opcionales"), OPTIONAL_METRICS, "metricas_opcionales", OPTIONAL_METRICS)
            validar_valores_neutros(drop, opcionales)
            resultado = probabilidades_campeon(recepcion, drop_metricas=drop, metricas_opcionales=opcionales,
                                               reglamento=datos.get("reglamento"))
        except (ReglamentoInvalido, ValueError) as error:
            return Response({"detail": str(error)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(resultado)
