from rest_framework import serializers


class PrediccionSerializer(serializers.Serializer):
    """Misma entrada que el boton "Predecir" de la web"""

    recepcion = serializers.JSONField(help_text="Lista [anio, fase, ...partidos] que envia la app")
    reglamento = serializers.JSONField(required=False, allow_null=True,
                                       help_text="Reglamento de mejores terceros personalizado (opcional)")
    drop_metricas = serializers.ListField(child=serializers.CharField(), required=False,
                                          help_text="Metricas base que el usuario descarta")
    metricas_opcionales = serializers.ListField(child=serializers.CharField(), required=False,
                                                help_text="Metricas opcionales que el usuario acepta")

    def validate_recepcion(self, valor):
        if not isinstance(valor, list) or len(valor) < 2:
            raise serializers.ValidationError("recepcion debe ser una lista con al menos [anio, fase]")
        return valor
