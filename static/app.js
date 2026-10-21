// Sends the message to POST /generate and shows the answer on the page.
// Model output is inserted with textContent, never innerHTML, so text from
// the model or the customer can't run as code in the page.

const SAMPLES = {
  charge:
    "This is the second time you've charged my card twice for the same order (#48213). " +
    "I want my money back today or I'm cancelling my subscription.",
  package:
    "My package was marked as delivered yesterday, but it is not at my door and my " +
    "neighbours have not seen it. Tracking number 1Z999AA10123456784.",
  happy:
    "The new app update is great, the checkout is much faster now! " +
    "Is there a way to save two delivery addresses?",
};

const $ = (id) => document.getElementById(id);
const form = $("ticket-form");
const message = $("message");

// Show exactly one of: empty hint, loading text, error box, result.
function show(state) {
  for (const id of ["empty", "loading", "error", "result"]) {
    $(id).hidden = id !== state;
  }
}

function sentimentStyle(score) {
  if (score < 35) return { label: "Negative", color: "var(--negative)" };
  if (score <= 65) return { label: "Neutral", color: "var(--neutral)" };
  return { label: "Positive", color: "var(--positive)" };
}

function showResult(data) {
  const style = sentimentStyle(data.sentiment);
  $("sentiment-value").textContent = data.sentiment;
  $("sentiment-label").textContent = style.label;
  $("sentiment-bar").style.width = `${data.sentiment}%`;
  $("sentiment-bar").style.background = style.color;
  $("summary").textContent = data.summary;
  $("response").textContent = data.response;
  $("next-action").textContent = data.next_action;
  $("meta").textContent = `${data.model} · ${data.duration} s`;
  show("result");
}

function showError(text) {
  $("error").textContent = text;
  show("error");
}

async function analyze(event) {
  event.preventDefault();
  const model = form.elements.model.value;
  const button = $("submit");

  button.disabled = true;
  $("loading").textContent = `${model[0].toUpperCase() + model.slice(1)} is reading the message…`;
  show("loading");

  try {
    const reply = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message.value, model }),
    });
    const data = await reply.json();
    if (reply.ok) {
      showResult(data);
    } else {
      showError(data.error || `Request failed (${reply.status}).`);
    }
  } catch {
    showError("Can't reach the app. Is the Flask server still running?");
  } finally {
    button.disabled = false;
  }
}

function updateCounter() {
  $("counter").textContent = `${message.value.length} / 4000`;
}

form.addEventListener("submit", analyze);
message.addEventListener("input", updateCounter);

// Ctrl + Enter (Cmd + Enter on Mac) submits the form.
message.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});

document.querySelectorAll("[data-sample]").forEach((chip) => {
  chip.addEventListener("click", () => {
    message.value = SAMPLES[chip.dataset.sample];
    updateCounter();
    message.focus();
  });
});

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("response").textContent);
    $("copy").textContent = "Copied";
    setTimeout(() => ($("copy").textContent = "Copy"), 1500);
  } catch {
    $("copy").textContent = "Copy failed";
  }
});
