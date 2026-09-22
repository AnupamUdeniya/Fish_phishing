console.log("Content script running");

function getEmailData() {
    // extracting subject, sender, and body from the email you wish to scan
    const subject = document.querySelector("h2.hP")?.innerText || "";
    const sender = document.querySelector("span.gD")?.getAttribute("email") || "";
    const body = document.querySelector("div.a3s")?.innerText || "";
  
    if (!body) return;
  
    const emailData = {
      sender,
      subject,
      body
    };

    try {
      chrome.runtime.sendMessage({
        type: "EMAIL_DATA",
        payload: emailData
      });
    } catch (error) {
      console.warn("Failed to send message: ", error);
    }
  
    console.log("Extracted email data: ", emailData);
  }
  
  // use mutation observer to detect changes in the DOM structure
  let timeoutId;

  const observer = new MutationObserver(() => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      const subject = document.querySelector("h2.hP");
      const sender = document.querySelector("span.gD");
      const bodyEl = document.querySelector("div.a3s");

      if (subject && sender && bodyEl && bodyEl.innerText.trim().length > 50) {
        getEmailData();
      }
    }, 500); // wait 500ms before running getEmailData
  });
  
  observer.observe(document.body, { childList: true, subtree: true });
