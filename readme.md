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

## Intalação via Docker

Para utilizar o serviço via Docker, é necessário ter o Docker instalado na máquina.

Primeiro, é preciso garantir a existência da network `psitest`.

```bash
docker network create psitest
```

Em seguida, execute o comando para criar a imagem e o container do serviço.

```bash
docker compose up
```

O serviço estará disponível em `http://localhost:8000`.
O MongoDB estará disponível em `http://localhost:27017`.
