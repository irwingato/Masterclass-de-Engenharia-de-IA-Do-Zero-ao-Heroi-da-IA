@echo off
:: ----------------------------------------------------------------------------
:: Script de execução do dat1_tensorflow.py com supressão de logs do TensorFlow
:: ----------------------------------------------------------------------------
::
:: PROBLEMA:
::   Ao rodar `python dat1_tensorflow.py` no CMD, aparecem mensagens como:
::     WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
::     I0000 ... port.cc:153] oneDNN custom operations are on...
::   Essas mensagens são geradas pelo runtime C++ do TensorFlow (absl / oneDNN)
::   e saem diretamente no STDERR do Windows ANTES do interpretador Python
::   assumir o controle do console. Por isso, comandos Python sozinhos
::   (ex: contextlib.redirect_stderr) nem sempre conseguem silenciá-las.
::
:: SOLUÇÃO:
::   Definir as variáveis de ambiente no nível do SISTEMA (CMD) ANTES de
::   iniciar o Python. Assim, o TensorFlow as lê durante sua própria
::   inicialização C++ e não emite os avisos.
::
:: ----------------------------------------------------------------------------
:: Variáveis de ambiente e o que cada uma faz:
:: ----------------------------------------------------------------------------
::
:: set TF_CPP_MIN_LOG_LEVEL=3
::   Controla o nível de log do runtime C++ do TensorFlow.
::   Níveis:
::     0 = exibe todos os logs (padrão)
::     1 = filtra mensagens INFO
::     2 = filtra INFO + WARNING
::     3 = filtra INFO + WARNING + ERROR (apenas FATAL permanece)
::   Com 3, removemos a maior parte do ruído de console do TensorFlow.
::
:: set TF_ENABLE_ONEDNN_OPTS=0
::   Desativa as otimizações oneDNN (Intel oneAPI Deep Neural Network Library).
::   Quando ativas, o TensorFlow avisa que resultados numéricos podem variar
::   levemente devido à ordem de operações de ponto flutuante. Desligando,
::   eliminamos o aviso do port.cc:153.
::
:: set AUTOGRAPH_VERBOSITY=0
::   Controla a verbosidade do AutoGraph (ferramenta que converte código Python
::   puro em graphs do TensorFlow). '0' suprime mensagens de debug.
::
:: ----------------------------------------------------------------------------
:: Execução do Python
:: ----------------------------------------------------------------------------
::
:: python dat1_tensorflow.py
::   Inicia o interpretador Python e roda o script. Como as variáveis acima
::   já estão definidas no ambiente do CMD, o TensorFlow as herda e não
::   imprime os avisos indesejados.
::
:: pause
::   Mantém a janela aberta após o término para você conferir a saída.
::   Remova se não precisar.
:: ----------------------------------------------------------------------------

set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
set AUTOGRAPH_VERBOSITY=0

python dat1_tensorflow.py

pause
