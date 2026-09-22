console.log("Content script running");

function getEmailData() {
    const subject = document.querySelector("h2.hP")?.innerText || "";
    const sender = document.querySelector("span.gD")?.getAttribute("email") || "";
    const body = document.querySelector("div.a3s")?.innerText || "";

    return {
      sender,
      subject,
      body
    };
}

function sendEmailData() {
    const emailData = getEmailData();

    try {
      chrome.runtime.sendMessage({
        type: "EMAIL_DATA",
        payload: emailData
      });
    } catch (error) {
      console.warn("Failed to send message: ", error);
    }

    console.log("Extracted email data: ", emailData);
    return emailData;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request?.type !== "GET_EMAIL_DATA") {
    return false;
  }

  sendResponse({ payload: sendEmailData() });
  return true;
});

let timeoutId;

const observer = new MutationObserver(() => {
  clearTimeout(timeoutId);
  timeoutId = setTimeout(() => {
    const emailData = getEmailData();

    if (emailData.body.trim()) {
      sendEmailData();
    }
  }, 500);
});

if (document.body) {
  observer.observe(document.body, { childList: true, subtree: true });
}
