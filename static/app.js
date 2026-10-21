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

// Show exactly one of: empty hint, loading text, error box, result, comparison.
function show(state) {
  for (const id of ["empty", "loading", "error", "result", "comparison"]) {
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

// Small helper: create an element with a class and text (never HTML).
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const capitalize = (name) => name[0].toUpperCase() + name.slice(1);

function showComparison(results) {
  const box = $("comparison");
  box.replaceChildren();
  for (const item of results) {
    const card = el("article", "card");
    const head = el("div", "row between");
    head.append(el("h3", "", capitalize(item.model)));

    if (item.status === 200) {
      const style = sentimentStyle(item.sentiment);
      const badge = el("span", "badge", `${item.sentiment} · ${style.label}`);
      badge.style.background = style.color;
      head.append(badge);
      card.append(
        head,
        el("p", "muted", `${item.summary} (${item.duration} s)`),
        el("p", "reply", item.response),
        el("p", "action", item.next_action),
      );
    } else {
      card.append(head, el("p", "alert", item.error));
    }
    box.append(card);
  }
  show("comparison");
}

function showError(text) {
  $("error").textContent = text;
  show("error");
}

// POST a JSON body and show the answer with `onSuccess`, or the error text.
async function send(url, body, loadingText, onSuccess) {
  const buttons = [$("submit"), $("compare-button")];
  buttons.forEach((button) => (button.disabled = true));
  $("loading").textContent = loadingText;
  show("loading");

  try {
    const reply = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await reply.json();
    if (reply.ok) {
      onSuccess(data);
    } else {
      showError(data.error || `Request failed (${reply.status}).`);
    }
  } catch {
    showError("Can't reach the app. Is the Flask server still running?");
  } finally {
    buttons.forEach((button) => (button.disabled = false));
  }
}

function analyze(event) {
  event.preventDefault();
  const model = form.elements.model.value;
  send(
    "/generate",
    { message: message.value, model },
    `${capitalize(model)} is reading the message…`,
    showResult,
  );
}

function compareAll() {
  if (!form.reportValidity()) return; // same "message required" check as Analyze
  send(
    "/compare",
    { message: message.value },
    "Every model is reading the message, one after another…",
    (data) => showComparison(data.results),
  );
}

function updateCounter() {
  $("counter").textContent = `${message.value.length} / 4000`;
}

form.addEventListener("submit", analyze);
$("compare-button").addEventListener("click", compareAll);
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
