import json
import re
from typing import Any, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


from pydantic import BaseModel, Field

class PredRequest(BaseModel):
    valor_emprestimo: int = Field(
        default=10000, 
        description="Valor do empréstimo em R$, informado pelo usuário"
    )
    prazo_emprestimo_anos: int = Field(
        default=5, 
        description="Prazo do empréstimo em anos, informado pelo usuário"
    )
    faixa_idade: int = Field(
        default=30, 
        description="Idade do usuário em anos (ex: 27, 30)"
    )
    outros_planos_financiamento: int = Field(
        default=0,
        description="0 -> Não possui outros empréstimos; 1 -> Possui outros empréstimos (banco, loja ou outros)"
    )
    historico_credito: int = Field(
        default=0,
        description=(
            "0 -> Conta crítica / outros créditos existentes (não neste banco) | "
            "0 -> Atraso no pagamento de dívidas no passado | "
            "1 -> Todos os créditos pagos em dia até agora | "
            "1 -> Nenhum crédito tomado / todos pagos corretamente | "
            "1 -> Todos os créditos neste banco pagos corretamente"
        )
    )
    propriedade: int = Field(
        default=0,
        description=(
            "0 -> Desconhecido / sem propriedade | "
            "1 -> Poupança em sociedade / seguro de vida | "
            "1 -> Carro | "
            "2 -> Imóvel próprio"
        )
    )
    tempo_emprego_atual: int = Field(
        default=1,
        description=(
            "0 -> Desempregado ou menos de 1 ano | "
            "1 -> 1 a 4 anos | "
            "1 -> 4 a 7 anos | "
            "1 -> 7 anos ou mais"
        )
    )
    reserva_cc: int = Field(
        default=1,
        description="0 -> Até R$ 1600 | 1 -> Mais de R$ 1600"
    )
    tipo_residencia: int = Field(
        default=1,
        description="1 -> Própria | 0 -> Alugada"
    )
    conta_corrente: int = Field(
        default=0,
        description="0 -> Sem conta corrente / conta negativada | 1 -> Conta corrente positiva"
    )

    @field_validator("faixa_idade")
    def ajuste_idade_faixa(cls, v):
        # Remove o último dígito (ex: 30 -> 3, 27 -> 2)
        return int(str(v)[:-1]) if v >= 10 else 0


device_list_responses = {
    200: {
        "description": "List of all registered device names.",
        "content": {"application/json": {"example": ["device_01", "device_02"]}},
    }
}
