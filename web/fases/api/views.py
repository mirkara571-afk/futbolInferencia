from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from fases.modulo import ReglamentoInvalido
from fases.views import _respuesta_json, predecir_recepcion

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
    """POST /api/probabilidades/: reservado para la Etapa G (Montecarlo de campeones)"""

    @extend_schema(request=PrediccionSerializer, responses={200: OpenApiTypes.OBJECT},
                   description="Probabilidades de campeon (Etapa G; aun no implementado)")
    def post(self, request):
        return Response({"detail": "La Etapa G (Montecarlo de campeones) aun no esta implementada."},
                        status=status.HTTP_501_NOT_IMPLEMENTED)
