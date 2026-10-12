# Projeto - Análise de fraude em números de cartão de crédito
## Problema:
<p>
    Em um conjunto de números de cartão podem existir aqueles que são fraudes.
</p>
<p>
    A solução é usar métodos que possam coletar esses dados malicioso, porém, outro problema é o desbalanceamento, que pode alterar a "realidade", apresentando dados nada precisos.
</p>

## Preparação dos dados:
<p>
    Na missão de encontrar o máximo de fraudes com a melhor precisão, antes de usar os métodos, os dados passaram por um pré-processamento (com auxílio de bibliotecas especiais) para alcançar os devidos objetivos...
    <p>
        <ul>
            <li>
                NumPy: Realizou um cálculo de logaritmo para reduzir a distorção de dados (discrepância entre valores de compras); 
            </li>
                model_selection: Divisão estratégica e segura dos dados. Garante que o modelo mantenha a proporção de fraudes idêntica tanto na base quanto no teste;
            <li>
                preprocessing: Ajusta a escala para igualar a importância das variáveis. Pode ocorrer do modelo considerar a coluna com números maiores a mais importante. Logo, para manter a mesma importância de grandeza, a variável é reajustada para média 0 e desvio padrão 1, assim garante que fiquem na mesma "categoria".
            </li>
        </ul>
    </p>
</p>

## Comparação - em andamento

## Limiar de decisão - em andamento