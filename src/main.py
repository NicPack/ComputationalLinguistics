import tiktoken
import torch
from tqdm import tqdm

from config import Settings
from data import estimate_loss, get_batch
from models import GPTLanguageModel, LSTMLanguageModel

if __name__ == "__main__":
    settings = Settings()
    model_name = settings.model_name.lower()
    encoding = tiktoken.encoding_for_model("gpt-4o")

    with open("datasets/high_quality_plwiki.txt", "r", encoding="utf-8") as f:
        text = f.read()

    PATH = f"checkpoints/{model_name}.pt"

    torch.manual_seed(1337)

    # Train and test splits
    data = torch.tensor(encoding.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    if model_name.startswith("gpt"):
        model = GPTLanguageModel(
            n_head=settings.n_head,
            block_size=settings.block_size,
            n_embd=settings.n_embd,
            n_layer=settings.n_layer,
            vocab_size=settings.vocab_size,
            dropout=settings.dropout,
            device=settings.device,
        )
        model.load_state_dict(
            torch.load(PATH, weights_only=True, map_location=torch.device("mps"))
        )
    elif model_name.startswith("lstm"):
        model = LSTMLanguageModel(
            n_layer=settings.n_layer,
            block_size=settings.block_size,
            n_embd=settings.n_embd,
            vocab_size=settings.vocab_size,
            dropout=settings.dropout,
            device=settings.device,
        )
    else:
        print("wrong model selected")
        quit

    m = model.to(settings.device)

    m = model.to(settings.device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = settings.optimizer(model.parameters(), lr=settings.learning_rate)

    print("training started:\n\n")
    for iter in tqdm(range(settings.num_epochs)):
        # every once in a while evaluate the loss on train and val sets
        if iter % settings.eval_interval == 0 or iter == settings.num_epochs - 1:
            losses = estimate_loss(
                model=model,
                eval_iters=settings.eval_iters,
                block_size=settings.block_size,
                batch_size=settings.batch_size,
                train_data=train_data,
                val_data=val_data,
                device=settings.device,
            )
            print(
                f"step {iter}: train loss {losses['train']:.4f}, training perplexity: {torch.exp(losses['train']):.4f}",
                f"val loss {losses['val']:.4f}, val perplexity: {torch.exp(losses['val']):.4f}",
                sep="\n",
            )
            torch.save(model.state_dict(), f=PATH)

            # generate from the model
            context = torch.zeros((1, 1), dtype=torch.long, device=settings.device)
            generated_text = encoding.decode(
                m.generate(context, max_new_tokens=50)[0].tolist()
            )
            print(f"Epoch: {iter}", f"Generated text: {generated_text}", sep="\n")

        # sample a batch of data
        xb, yb = get_batch(
            block_size=settings.block_size,
            batch_size=settings.batch_size,
            data=train_data,
            device=settings.device,
        )

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=settings.device)
    print(encoding.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
