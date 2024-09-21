# Psitest-Imagem

## Descrição

O projeto `psitest-imagens` é uma ferramenta extrair dados sobre as respostas escolhidas em um teste múltipla-escolha, utilizando técnicas de aprendizado de máquina e visão computacional.

## Instalação local

Para utilizar o serviço localmente, é recomendado a criação de um ambiente virtual.

```bash
python3 -m venv .venv
ssource .venv/bin/activate
```

Após a criação do ambiente virtual, instale as dependências do projeto.

```bash
pip install -r requirements.txt
```

### Execução

Antes de executar o servidor, crie um arquivo `.env` na raiz do projeto com as seguintes variáveis de ambiente:

```
MONGODB_URL=localhost:27017
```

A URL precisa apontar para uma porta que tenha um servidor MongoDB em execução.

Para executar o servidor, utilize o comando:

```bash
fastapi run app --port 8000
```

O servidor estará disponível em `http://localhost:8000`.
